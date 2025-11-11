//! Application runner coordinating crawl execution modes.

use crate::agents::{
    AgentRegistry, CrawlTask, InlineString, DEFAULT_AGENT_CAPACITY, DEFAULT_QUEUE_DEPTH,
};
use crate::frontier::{Frontier, FrontierError, DEFAULT_FRONTIER_QUEUE, DEFAULT_FRONTIER_SEEN};
#[cfg(not(feature = "multi_thread"))]
use crate::html::{stream_links, HtmlStreamError};
use crate::{Cli, CrawlControls};
use futures_util::future::join_all;
use reqwest::Client;
#[cfg(feature = "multi_thread")]
use reqwest::Response;
#[cfg(feature = "multi_thread")]
use scraper::{Html, Selector};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::runtime::Builder;
use tokio::task::yield_now;
#[cfg(not(feature = "multi_thread"))]
use tokio::task::{spawn_local, LocalSet};
#[cfg(feature = "multi_thread")]
use tokio::sync::mpsc;
use tokio::time::sleep;
use url::Url;

const USER_AGENT: &str = "fastcrawl-example/0.1 (+https://github.com/aaronlifton/fastcrawl)";
const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(100);
#[cfg(feature = "multi_thread")]
const LINK_BATCH_CHANNEL_CAPACITY: usize = DEFAULT_AGENT_CAPACITY * 4;

/// Predicate used to accept or reject discovered URLs.
pub type UrlFilter = Arc<dyn Fn(&Url) -> bool + Send + Sync>;

/// Entry point used by examples to run the crawler with the provided seed URLs and link filter.
pub fn run(cli: Cli, seeds: &[&str], filter: UrlFilter) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "multi_thread")]
    {
        let rt = Builder::new_multi_thread().enable_all().build()?;
        rt.block_on(run_multi_thread(cli, seeds, Arc::clone(&filter)))?
    }
    #[cfg(not(feature = "multi_thread"))]
    {
        let rt = Builder::new_current_thread().enable_all().build()?;
        let local = LocalSet::new();
        let cli_owned = cli.clone();
        rt.block_on(local.run_until(run_single_thread(cli_owned, seeds, filter)))?;
    }
    Ok(())
}

struct AppState {
    registry: Arc<AgentRegistry<DEFAULT_AGENT_CAPACITY, DEFAULT_QUEUE_DEPTH>>,
    frontier: Arc<Frontier<DEFAULT_FRONTIER_QUEUE, DEFAULT_FRONTIER_SEEN>>,
    stop_requested: Arc<AtomicBool>,
    active_tasks: Arc<AtomicUsize>,
    metrics: Arc<Metrics>,
    controls: Arc<CrawlControls>,
    client: Client,
    run_duration: Duration,
    link_filter: UrlFilter,
}

impl AppState {
    fn new(cli: &Cli, filter: UrlFilter) -> Result<Self, reqwest::Error> {
        let registry =
            Arc::new(AgentRegistry::<DEFAULT_AGENT_CAPACITY, DEFAULT_QUEUE_DEPTH>::new());
        let frontier = Arc::new(Frontier::<DEFAULT_FRONTIER_QUEUE, DEFAULT_FRONTIER_SEEN>::new());
        let stop_requested = Arc::new(AtomicBool::new(false));
        let active_tasks = Arc::new(AtomicUsize::new(0));
        let metrics = Arc::new(Metrics::default());
        let controls = Arc::new(cli.build_controls());
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .redirect(reqwest::redirect::Policy::limited(5))
            .timeout(Duration::from_secs(10))
            .build()?;

        Ok(Self {
            registry,
            frontier,
            stop_requested,
            active_tasks,
            metrics,
            controls,
            client,
            run_duration: cli.run_duration(),
            link_filter: filter,
        })
    }
}

#[cfg(not(feature = "multi_thread"))]
async fn run_single_thread(
    cli: Cli,
    seeds: &[&str],
    filter: UrlFilter,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = AppState::new(&cli, filter)?;
    seed_frontier(state.frontier.as_ref(), seeds).await;
    let start = Instant::now();

    let dispatcher = {
        let frontier = Arc::clone(&state.frontier);
        let registry = Arc::clone(&state.registry);
        spawn_local(async move {
            frontier.dispatch_loop(registry).await;
        })
    };

    let mut workers = Vec::new();
    for id in 0..DEFAULT_AGENT_CAPACITY {
        workers.push(spawn_streaming_worker(&state, id));
    }

    finish_run(state, dispatcher, workers, start).await;
    Ok(())
}

#[cfg(feature = "multi_thread")]
async fn run_multi_thread(
    cli: Cli,
    seeds: &[&str],
    filter: UrlFilter,
) -> Result<(), Box<dyn std::error::Error>> {
    let state = AppState::new(&cli, filter)?;
    seed_frontier(state.frontier.as_ref(), seeds).await;
    let start = Instant::now();

    let dispatcher = {
        let frontier = Arc::clone(&state.frontier);
        let registry = Arc::clone(&state.registry);
        tokio::spawn(async move {
            frontier.dispatch_loop(registry).await;
        })
    };

    let (link_tx, link_rx) = mpsc::channel(LINK_BATCH_CHANNEL_CAPACITY);
    let enqueue_handle = {
        let frontier = Arc::clone(&state.frontier);
        let stop_requested = Arc::clone(&state.stop_requested);
        let controls = Arc::clone(&state.controls);
        let metrics = Arc::clone(&state.metrics);
        let filter = Arc::clone(&state.link_filter);
        tokio::spawn(async move {
            link_enqueue_loop(
                link_rx,
                frontier,
                stop_requested,
                controls,
                metrics,
                filter,
            )
            .await;
        })
    };

    let mut workers = Vec::new();
    for id in 0..DEFAULT_AGENT_CAPACITY {
        workers.push(spawn_buffered_worker(&state, id, link_tx.clone()));
    }
    drop(link_tx);
    workers.push(enqueue_handle);

    finish_run(state, dispatcher, workers, start).await;
    Ok(())
}

async fn finish_run(
    state: AppState,
    dispatcher: tokio::task::JoinHandle<()>,
    workers: Vec<tokio::task::JoinHandle<()>>,
    start: Instant,
) {
    sleep(state.run_duration).await;
    state.stop_requested.store(true, Ordering::Release);
    wait_for_drain(
        state.registry.as_ref(),
        state.frontier.as_ref(),
        state.active_tasks.as_ref(),
    )
    .await;
    state.frontier.shutdown();
    for agent in state.registry.iter() {
        agent.shutdown();
    }

    let _ = dispatcher.await;
    join_all(workers).await;
    state.metrics.report(start.elapsed());
}

#[cfg(not(feature = "multi_thread"))]
fn spawn_streaming_worker(state: &AppState, id: usize) -> tokio::task::JoinHandle<()> {
    let registry = Arc::clone(&state.registry);
    let frontier = Arc::clone(&state.frontier);
    let client = state.client.clone();
    let stop_requested = Arc::clone(&state.stop_requested);
    let active_tasks = Arc::clone(&state.active_tasks);
    let metrics = Arc::clone(&state.metrics);
    let controls = Arc::clone(&state.controls);
    let filter = Arc::clone(&state.link_filter);
    spawn_local(async move {
        worker_loop_streaming(
            id,
            registry,
            frontier,
            client,
            stop_requested,
            active_tasks,
            controls,
            metrics,
            filter,
        )
        .await;
    })
}

#[cfg(feature = "multi_thread")]
fn spawn_buffered_worker(
    state: &AppState,
    id: usize,
    link_tx: mpsc::Sender<LinkBatch>,
) -> tokio::task::JoinHandle<()> {
    let registry = Arc::clone(&state.registry);
    let client = state.client.clone();
    let stop_requested = Arc::clone(&state.stop_requested);
    let active_tasks = Arc::clone(&state.active_tasks);
    let metrics = Arc::clone(&state.metrics);
    let controls = Arc::clone(&state.controls);
    let filter = Arc::clone(&state.link_filter);
    tokio::spawn(async move {
        worker_loop_buffered(
            id,
            registry,
            client,
            stop_requested,
            active_tasks,
            controls,
            metrics,
            filter,
            link_tx,
        )
        .await;
    })
}

#[cfg(not(feature = "multi_thread"))]
async fn worker_loop_streaming<
    const COUNT: usize,
    const DEPTH: usize,
    const QUEUE: usize,
    const SEEN: usize,
>(
    agent_id: usize,
    registry: Arc<AgentRegistry<COUNT, DEPTH>>,
    frontier: Arc<Frontier<QUEUE, SEEN>>,
    client: Client,
    stop_requested: Arc<AtomicBool>,
    active_tasks: Arc<AtomicUsize>,
    controls: Arc<CrawlControls>,
    metrics: Arc<Metrics>,
    filter: UrlFilter,
) {
    let Some(agent) = registry.agent(agent_id) else {
        eprintln!("worker {agent_id}: agent missing");
        return;
    };

    while let Some(task) = agent.next_task().await {
        let _active = ActiveTaskGuard::new(active_tasks.as_ref());
        if let Err(err) = handle_task_streaming(
            &client,
            Arc::clone(&frontier),
            task,
            stop_requested.as_ref(),
            Arc::clone(&controls),
            metrics.as_ref(),
            Arc::clone(&filter),
        )
        .await
        {
            let TaskError { url, message, kind } = err;
            eprintln!(
                "worker {agent_id}: failed {url} -> {err}",
                url = url.unwrap_or_default(),
                err = message
            );
            metrics.record_error(kind);
        }
    }
}

#[cfg(feature = "multi_thread")]
async fn worker_loop_buffered<const COUNT: usize, const DEPTH: usize>(
    agent_id: usize,
    registry: Arc<AgentRegistry<COUNT, DEPTH>>,
    client: Client,
    stop_requested: Arc<AtomicBool>,
    active_tasks: Arc<AtomicUsize>,
    controls: Arc<CrawlControls>,
    metrics: Arc<Metrics>,
    filter: UrlFilter,
    link_tx: mpsc::Sender<LinkBatch>,
) {
    let Some(agent) = registry.agent(agent_id) else {
        eprintln!("worker {agent_id}: agent missing");
        return;
    };

    while let Some(task) = agent.next_task().await {
        let _active = ActiveTaskGuard::new(active_tasks.as_ref());
        if let Err(err) = handle_task_buffered(
            &client,
            task,
            stop_requested.as_ref(),
            Arc::clone(&controls),
            metrics.as_ref(),
            Arc::clone(&filter),
            &link_tx,
        )
        .await
        {
            let TaskError { url, message, kind } = err;
            eprintln!(
                "worker {agent_id}: failed {url} -> {err}",
                url = url.unwrap_or_default(),
                err = message
            );
            metrics.record_error(kind);
        }
    }
}

#[cfg(not(feature = "multi_thread"))]
async fn handle_task_streaming<const QUEUE: usize, const SEEN: usize>(
    client: &Client,
    frontier: Arc<Frontier<QUEUE, SEEN>>,
    task: CrawlTask,
    stop_requested: &AtomicBool,
    controls: Arc<CrawlControls>,
    metrics: &Metrics,
    filter: UrlFilter,
) -> Result<(), TaskError> {
    let url = task.url().to_string();
    println!("[depth {}] {}", task.depth(), url);

    if task.politeness_delay_ms() > 0 {
        sleep(Duration::from_millis(task.politeness_delay_ms() as u64)).await;
    }

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|err| TaskError::http(&url, err))?;

    if !response.status().is_success() {
        return Ok(());
    }

    metrics.record_page_fetched();

    let base = Url::parse(&url).map_err(|err| TaskError::parse(&url, err))?;
    let base_for_links = Arc::new(base.clone());
    let controls_for_links = Arc::clone(&controls);
    let filter_for_links = Arc::clone(&filter);
    let discovered_links = stream_links(response, controls.max_links_per_page(), move |href| {
        base_for_links.join(href).ok().filter(|candidate| {
            candidate
                .domain()
                .map(|domain| controls_for_links.is_domain_allowed(domain))
                .unwrap_or(false)
                && (filter_for_links.as_ref())(candidate)
        })
    })
    .await
    .map_err(|err| TaskError::html(&url, err))?;

    if task.depth() >= controls.max_depth() {
        return Ok(());
    }

    enqueue_discovered_links(
        discovered_links,
        task.depth(),
        frontier,
        stop_requested,
        controls,
        metrics,
        filter,
    )
    .await
}

#[cfg(feature = "multi_thread")]
async fn handle_task_buffered(
    client: &Client,
    task: CrawlTask,
    stop_requested: &AtomicBool,
    controls: Arc<CrawlControls>,
    metrics: &Metrics,
    filter: UrlFilter,
    link_tx: &mpsc::Sender<LinkBatch>,
) -> Result<(), TaskError> {
    let url = task.url().to_string();
    println!("[depth {}] {}", task.depth(), url);

    if task.politeness_delay_ms() > 0 {
        sleep(Duration::from_millis(task.politeness_delay_ms() as u64)).await;
    }

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|err| TaskError::http(&url, err))?;

    if !response.status().is_success() {
        return Ok(());
    }

    metrics.record_page_fetched();

    let base = Url::parse(&url).map_err(|err| TaskError::parse(&url, err))?;
    let discovered_links =
        extract_links_buffered(&url, &base, response, controls.as_ref(), &filter).await?;

    if task.depth() >= controls.max_depth() || discovered_links.is_empty() {
        return Ok(());
    }

    if link_tx
        .send(LinkBatch {
            depth: task.depth(),
            links: discovered_links,
        })
        .await
        .is_err()
    {
        if !stop_requested.load(Ordering::Acquire) {
            eprintln!("enqueue channel closed while processing {url}");
        }
    }

    Ok(())
}

async fn enqueue_discovered_links<const QUEUE: usize, const SEEN: usize>(
    discovered_links: Vec<Url>,
    parent_depth: u8,
    frontier: Arc<Frontier<QUEUE, SEEN>>,
    stop_requested: &AtomicBool,
    controls: Arc<CrawlControls>,
    metrics: &Metrics,
    filter: UrlFilter,
) -> Result<(), TaskError> {
    for discovered in discovered_links {
        if stop_requested.load(Ordering::Acquire) {
            break;
        }
        let domain_allowed = discovered
            .domain()
            .map(|domain| controls.is_domain_allowed(domain))
            .unwrap_or(false);
        if !domain_allowed || !(filter.as_ref())(&discovered) {
            continue;
        }
        metrics.record_url_discovered();
        let inline = match InlineString::try_from_str(discovered.as_str()) {
            Ok(value) => value,
            Err(err) => {
                eprintln!(
                    "skip {} due to inline alloc error: {err:?}",
                    discovered.as_str()
                );
                continue;
            }
        };
        let next = CrawlTask::new(inline, parent_depth + 1).with_delay(controls.politeness_delay());

        let mut task_to_enqueue = next;
        loop {
            match frontier.push_task(task_to_enqueue).await {
                Ok(()) => {
                    metrics.record_url_enqueued();
                    break;
                }
                Err(FrontierError::Duplicate(_)) => {
                    metrics.record_duplicate();
                    break;
                }
                Err(FrontierError::QueueFull(returned)) => {
                    task_to_enqueue = returned;
                    yield_now().await;
                }
                Err(err) => {
                    return Err(TaskError::frontier(discovered.as_str(), err));
                }
            }
        }
    }

	Ok(())
}

#[cfg(feature = "multi_thread")]
struct LinkBatch {
    depth: u8,
    links: Vec<Url>,
}

#[cfg(feature = "multi_thread")]
async fn link_enqueue_loop<const QUEUE: usize, const SEEN: usize>(
    mut rx: mpsc::Receiver<LinkBatch>,
    frontier: Arc<Frontier<QUEUE, SEEN>>,
    stop_requested: Arc<AtomicBool>,
    controls: Arc<CrawlControls>,
    metrics: Arc<Metrics>,
    filter: UrlFilter,
) {
    while let Some(batch) = rx.recv().await {
        if batch.links.is_empty() {
            continue;
        }

        if let Err(err) = enqueue_discovered_links(
            batch.links,
            batch.depth,
            Arc::clone(&frontier),
            stop_requested.as_ref(),
            Arc::clone(&controls),
            metrics.as_ref(),
            Arc::clone(&filter),
        )
        .await
        {
            let TaskError { url, message, .. } = err;
            let label = url.unwrap_or_else(|| "<unknown>".to_string());
            eprintln!("enqueue loop failed for {label}: {message}");
        }
    }
}

#[cfg(feature = "multi_thread")]
async fn extract_links_buffered(
    source_url: &str,
    base: &Url,
    response: Response,
    controls: &CrawlControls,
    filter: &UrlFilter,
) -> Result<Vec<Url>, TaskError> {
    let body = response
        .text()
        .await
        .map_err(|err| TaskError::http(source_url, err))?;

    let document = Html::parse_document(&body);
    let selector = Selector::parse("a[href]").expect("valid selector");
    let mut links = Vec::new();
    for element in document.select(&selector) {
        if let Some(href) = element.value().attr("href") {
            if let Ok(url) = base.join(href) {
                let domain_allowed = url
                    .domain()
                    .map(|domain| controls.is_domain_allowed(domain))
                    .unwrap_or(false);
                if domain_allowed && (filter.as_ref())(&url) {
                    links.push(url);
                    if links.len() >= controls.max_links_per_page() {
                        break;
                    }
                }
            }
        }
    }
    Ok(links)
}

async fn seed_frontier(
    frontier: &Frontier<DEFAULT_FRONTIER_QUEUE, DEFAULT_FRONTIER_SEEN>,
    seeds: &[&str],
) {
    for &seed in seeds {
        frontier
            .push_seed_url(seed, 0)
            .await
            .expect("seed enqueue succeeds");
    }
}

struct ActiveTaskGuard<'a> {
    counter: &'a AtomicUsize,
}

impl<'a> ActiveTaskGuard<'a> {
    fn new(counter: &'a AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self { counter }
    }
}

impl Drop for ActiveTaskGuard<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Default)]
struct Metrics {
    pages_fetched: AtomicUsize,
    urls_discovered: AtomicUsize,
    urls_enqueued: AtomicUsize,
    duplicates_filtered: AtomicUsize,
    frontier_rejections: AtomicUsize,
    http_errors: AtomicUsize,
    parse_errors: AtomicUsize,
}

impl Metrics {
    fn record_page_fetched(&self) {
        self.pages_fetched.fetch_add(1, Ordering::Relaxed);
    }

    fn record_url_discovered(&self) {
        self.urls_discovered.fetch_add(1, Ordering::Relaxed);
    }

    fn record_url_enqueued(&self) {
        self.urls_enqueued.fetch_add(1, Ordering::Relaxed);
    }

    fn record_duplicate(&self) {
        self.duplicates_filtered.fetch_add(1, Ordering::Relaxed);
    }

    fn record_error(&self, kind: TaskErrorKind) {
        match kind {
            TaskErrorKind::Http => {
                self.http_errors.fetch_add(1, Ordering::Relaxed);
            }
            TaskErrorKind::Parse => {
                self.parse_errors.fetch_add(1, Ordering::Relaxed);
            }
            TaskErrorKind::Frontier => {
                self.frontier_rejections.fetch_add(1, Ordering::Relaxed);
            }
            #[cfg(not(feature = "multi_thread"))]
            TaskErrorKind::Html => {
                self.parse_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn report(&self, elapsed: Duration) {
        let secs = elapsed.as_secs_f32().max(f32::EPSILON);
        let fetched = self.pages_fetched.load(Ordering::Relaxed);
        println!("--- crawl metrics ({secs:.2}s) ---");
        println!("pages fetched: {}", fetched);
        println!("urls fetched/sec: {:.2}", fetched as f32 / secs);
        println!(
            "urls discovered: {}",
            self.urls_discovered.load(Ordering::Relaxed)
        );
        println!(
            "urls enqueued: {}",
            self.urls_enqueued.load(Ordering::Relaxed)
        );
        println!(
            "duplicate skips: {}",
            self.duplicates_filtered.load(Ordering::Relaxed)
        );
        println!(
            "frontier rejects: {}",
            self.frontier_rejections.load(Ordering::Relaxed)
        );
        println!("http errors: {}", self.http_errors.load(Ordering::Relaxed));
        println!(
            "url parse errors: {}",
            self.parse_errors.load(Ordering::Relaxed)
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskErrorKind {
    Http,
    Parse,
    Frontier,
    #[cfg(not(feature = "multi_thread"))]
    Html,
}

struct TaskError {
    url: Option<String>,
    message: String,
    kind: TaskErrorKind,
}

impl TaskError {
    fn http(url: &str, err: reqwest::Error) -> Self {
        Self {
            url: Some(url.to_string()),
            message: format!("http error: {err}"),
            kind: TaskErrorKind::Http,
        }
    }

    fn parse(url: &str, err: url::ParseError) -> Self {
        Self {
            url: Some(url.to_string()),
            message: format!("url parse error: {err}"),
            kind: TaskErrorKind::Parse,
        }
    }

    fn frontier(url: &str, err: FrontierError) -> Self {
        Self {
            url: Some(url.to_string()),
            message: format!("frontier rejected: {err:?}"),
            kind: TaskErrorKind::Frontier,
        }
    }

    #[cfg(not(feature = "multi_thread"))]
    fn html(url: &str, err: HtmlStreamError) -> Self {
        Self {
            url: Some(url.to_string()),
            message: format!("html rewrite error: {err}"),
            kind: TaskErrorKind::Html,
        }
    }
}

async fn wait_for_drain<
    const COUNT: usize,
    const DEPTH: usize,
    const QUEUE: usize,
    const SEEN: usize,
>(
    registry: &AgentRegistry<COUNT, DEPTH>,
    frontier: &Frontier<QUEUE, SEEN>,
    active_tasks: &AtomicUsize,
) {
    loop {
        let workers_idle = active_tasks.load(Ordering::Acquire) == 0;
        let agents_idle = registry.iter().all(|agent| agent.backlog() == 0);
        let frontier_idle = frontier.pending() == 0;

        if workers_idle && agents_idle && frontier_idle {
            break;
        }

        sleep(DRAIN_POLL_INTERVAL).await;
    }
}
