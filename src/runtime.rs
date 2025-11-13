//! Application runner coordinating crawl execution modes.

use crate::agents::{
    AgentRegistry, CrawlTask, InlineString, DEFAULT_AGENT_CAPACITY, DEFAULT_QUEUE_DEPTH,
};
#[cfg(feature = "multi_thread")]
use crate::controls::PartitionStrategyArg;
use crate::frontier::{Frontier, FrontierError, DEFAULT_FRONTIER_QUEUE, DEFAULT_FRONTIER_SEEN};
use crate::html::{stream_links, HtmlStreamError};
use crate::{Cli, CrawlControls};
use futures_util::future::join_all;
use reqwest::Client;
#[cfg(feature = "multi_thread")]
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
#[cfg(feature = "multi_thread")]
use std::thread;
use std::time::{Duration, Instant};
#[cfg(feature = "multi_thread")]
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};
use tokio::runtime::Builder;
#[cfg(feature = "multi_thread")]
use tokio::sync::mpsc;
use tokio::task::{spawn_local, yield_now, LocalSet};
use tokio::time::sleep;
use url::Url;

const USER_AGENT: &str = "fastcrawl-example/0.1 (+https://github.com/aaronlifton/fastcrawl)";
const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(100);
#[cfg(feature = "multi_thread")]
const LINK_BATCH_CHANNEL_CAPACITY: usize = DEFAULT_AGENT_CAPACITY * 4;
#[cfg(feature = "multi_thread")]
const SHARD_STACK_SIZE: usize = 8 * 1024 * 1024;

/// Predicate used to accept or reject discovered URLs.
pub type UrlFilter = Arc<dyn Fn(&Url) -> bool + Send + Sync>;
type DynError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Clone)]
struct SharedRun {
    stop_requested: Arc<AtomicBool>,
    active_tasks: Arc<AtomicUsize>,
    metrics: Arc<Metrics>,
    controls: Arc<CrawlControls>,
    run_duration: Duration,
}

impl SharedRun {
    fn new(cli: &Cli) -> Self {
        Self {
            stop_requested: Arc::new(AtomicBool::new(false)),
            active_tasks: Arc::new(AtomicUsize::new(0)),
            metrics: Arc::new(Metrics::default()),
            controls: Arc::new(cli.build_controls()),
            run_duration: cli.run_duration(),
        }
    }
}

#[cfg(feature = "multi_thread")]
#[derive(Clone)]
struct ShardPartition {
    index: usize,
    strategy: Arc<ShardStrategy>,
}

#[cfg(feature = "multi_thread")]
impl ShardPartition {
    fn new(index: usize, strategy: Arc<ShardStrategy>) -> Self {
        Self { index, strategy }
    }

    fn shard_for_url(&self, url: &Url) -> usize {
        self.strategy.owner_for(url)
    }

    fn shard_for_str(&self, value: &str) -> usize {
        self.strategy.owner_for_str(value)
    }

    fn index(&self) -> usize {
        self.index
    }
}

#[cfg(feature = "multi_thread")]
#[derive(Clone)]
struct ShardStrategy {
    total: usize,
    kind: PartitionKind,
}

#[cfg(feature = "multi_thread")]
impl ShardStrategy {
    fn new(total: usize, kind: PartitionKind) -> Self {
        Self {
            total: total.max(1),
            kind,
        }
    }

    fn owner_for(&self, url: &Url) -> usize {
        match self.kind {
            PartitionKind::Hash => Self::hash(url.as_str(), self.total),
            PartitionKind::WikiPrefix => self
                .wiki_bucket(url)
                .unwrap_or_else(|| Self::hash(url.as_str(), self.total)),
        }
    }

    fn owner_for_str(&self, value: &str) -> usize {
        Url::parse(value)
            .ok()
            .map(|url| self.owner_for(&url))
            .unwrap_or_else(|| Self::hash(value, self.total))
    }

    fn wiki_bucket(&self, url: &Url) -> Option<usize> {
        if !url
            .domain()
            .map(|d| d.contains("wikipedia.org"))
            .unwrap_or(false)
        {
            return None;
        }
        let slug = url.path().strip_prefix("/wiki/")?;
        let first = slug.chars().next()?;
        if first.is_ascii_alphabetic() {
            let idx = (first.to_ascii_uppercase() as u8 - b'A') as usize;
            Some(idx % self.total)
        } else {
            None
        }
    }

    fn hash(value: &str, total: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        (hasher.finish() as usize) % total.max(1)
    }
}

#[cfg(feature = "multi_thread")]
#[derive(Clone, Copy)]
enum PartitionKind {
    Hash,
    WikiPrefix,
}

#[cfg(feature = "multi_thread")]
impl From<PartitionStrategyArg> for PartitionKind {
    fn from(value: PartitionStrategyArg) -> Self {
        match value {
            PartitionStrategyArg::Hash => PartitionKind::Hash,
            PartitionStrategyArg::WikiPrefix => PartitionKind::WikiPrefix,
        }
    }
}

/// Entry point used by examples to run the crawler with the provided seed URLs and link filter.
pub fn run(cli: Cli, seeds: &[&str], filter: UrlFilter) -> Result<(), DynError> {
    #[cfg(feature = "multi_thread")]
    {
        run_multi_thread(cli, seeds, Arc::clone(&filter))?;
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
    fn new_with_shared(filter: UrlFilter, shared: &SharedRun) -> Result<Self, reqwest::Error> {
        let registry =
            Arc::new(AgentRegistry::<DEFAULT_AGENT_CAPACITY, DEFAULT_QUEUE_DEPTH>::new());
        let frontier = Arc::new(Frontier::<DEFAULT_FRONTIER_QUEUE, DEFAULT_FRONTIER_SEEN>::new());
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .redirect(reqwest::redirect::Policy::limited(5))
            .timeout(Duration::from_secs(10))
            .build()?;

        Ok(Self {
            registry,
            frontier,
            stop_requested: Arc::clone(&shared.stop_requested),
            active_tasks: Arc::clone(&shared.active_tasks),
            metrics: Arc::clone(&shared.metrics),
            controls: Arc::clone(&shared.controls),
            client,
            run_duration: shared.run_duration,
            link_filter: filter,
        })
    }
}

#[cfg(not(feature = "multi_thread"))]
async fn run_single_thread(cli: Cli, seeds: &[&str], filter: UrlFilter) -> Result<(), DynError> {
    let shared = SharedRun::new(&cli);
    let state = AppState::new_with_shared(filter, &shared)?;
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

    finish_run(state, dispatcher, workers, start, true).await;
    Ok(())
}

#[cfg(feature = "multi_thread")]
fn run_multi_thread(cli: Cli, seeds: &[&str], filter: UrlFilter) -> Result<(), DynError> {
    let shard_count = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1);
    let shared = SharedRun::new(&cli);
    let partition_kind = PartitionKind::from(cli.partition_strategy());
    let strategy = Arc::new(ShardStrategy::new(shard_count, partition_kind));
    let seeds_owned: Vec<String> = seeds.iter().map(|s| s.to_string()).collect();
    let start = Instant::now();

    let mut receivers = Vec::with_capacity(shard_count);
    let mut senders_vec = Vec::with_capacity(shard_count);
    for _ in 0..shard_count {
        let (tx, rx) = mpsc::channel(LINK_BATCH_CHANNEL_CAPACITY);
        senders_vec.push(tx);
        receivers.push(rx);
    }
    let senders = Arc::new(senders_vec);

    let mut handles = Vec::with_capacity(shard_count);
    for (index, receiver) in receivers.into_iter().enumerate() {
        let shared_clone = shared.clone();
        let filter_clone = Arc::clone(&filter);
        let senders_clone = Arc::clone(&senders);
        let seeds_clone = seeds_owned.clone();
        let partition = ShardPartition::new(index, Arc::clone(&strategy));
        let builder = thread::Builder::new()
            .name(format!("fastcrawl-shard-{index}"))
            .stack_size(SHARD_STACK_SIZE);
        handles.push(builder.spawn(move || -> Result<(), DynError> {
            let rt = Builder::new_current_thread().enable_all().build()?;
            let local = LocalSet::new();
            rt.block_on(local.run_until(run_shard_streaming(
                seeds_clone,
                filter_clone,
                shared_clone,
                partition,
                senders_clone,
                receiver,
            )))
        })?);
    }

    // Drop the coordinator-owned sender references so shard inbox loops can observe channel closure.
    drop(senders);

    for handle in handles {
        handle.join().expect("shard thread panicked")?;
    }

    shared.metrics.report(start.elapsed());
    Ok(())
}

#[cfg(feature = "multi_thread")]
async fn run_shard_streaming(
    seeds: Vec<String>,
    filter: UrlFilter,
    shared: SharedRun,
    partition: ShardPartition,
    remotes: Arc<Vec<mpsc::Sender<LinkBatch>>>,
    inbox: mpsc::Receiver<LinkBatch>,
) -> Result<(), DynError> {
    let state = AppState::new_with_shared(filter, &shared)?;
    seed_partitioned_frontier(state.frontier.as_ref(), &seeds, &partition).await;
    let start = Instant::now();

    let dispatcher = {
        let frontier = Arc::clone(&state.frontier);
        let registry = Arc::clone(&state.registry);
        spawn_local(async move {
            frontier.dispatch_loop(registry).await;
        })
    };

    let inbox_handle = {
        let frontier = Arc::clone(&state.frontier);
        let stop_requested = Arc::clone(&state.stop_requested);
        let controls = Arc::clone(&state.controls);
        let metrics = Arc::clone(&state.metrics);
        let filter = Arc::clone(&state.link_filter);
        spawn_local(async move {
            shard_inbox_loop(inbox, frontier, stop_requested, controls, metrics, filter).await;
        })
    };

    let router = Arc::new(ShardRouter::new(partition, remotes));
    let mut workers = Vec::new();
    for id in 0..DEFAULT_AGENT_CAPACITY {
        workers.push(spawn_streaming_worker_sharded(
            &state,
            id,
            Arc::clone(&router),
        ));
    }
    workers.push(inbox_handle);

    finish_run(state, dispatcher, workers, start, false).await;
    Ok(())
}

async fn finish_run(
    state: AppState,
    dispatcher: tokio::task::JoinHandle<()>,
    workers: Vec<tokio::task::JoinHandle<()>>,
    start: Instant,
    report_metrics: bool,
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
    if report_metrics {
        state.metrics.report(start.elapsed());
    }
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
fn spawn_streaming_worker_sharded(
    state: &AppState,
    id: usize,
    router: Arc<ShardRouter>,
) -> tokio::task::JoinHandle<()> {
    let registry = Arc::clone(&state.registry);
    let frontier = Arc::clone(&state.frontier);
    let client = state.client.clone();
    let stop_requested = Arc::clone(&state.stop_requested);
    let active_tasks = Arc::clone(&state.active_tasks);
    let metrics = Arc::clone(&state.metrics);
    let controls = Arc::clone(&state.controls);
    let filter = Arc::clone(&state.link_filter);
    spawn_local(async move {
        worker_loop_streaming_sharded(
            id,
            registry,
            frontier,
            client,
            stop_requested,
            active_tasks,
            controls,
            metrics,
            filter,
            router,
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
async fn worker_loop_streaming_sharded<
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
    router: Arc<ShardRouter>,
) {
    let Some(agent) = registry.agent(agent_id) else {
        eprintln!("worker {agent_id}: agent missing");
        return;
    };

    while let Some(task) = agent.next_task().await {
        let _active = ActiveTaskGuard::new(active_tasks.as_ref());
        if let Err(err) = handle_task_streaming_sharded(
            &client,
            Arc::clone(&frontier),
            task,
            stop_requested.as_ref(),
            Arc::clone(&controls),
            metrics.as_ref(),
            Arc::clone(&filter),
            router.as_ref(),
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
async fn handle_task_streaming_sharded<const QUEUE: usize, const SEEN: usize>(
    client: &Client,
    frontier: Arc<Frontier<QUEUE, SEEN>>,
    task: CrawlTask,
    stop_requested: &AtomicBool,
    controls: Arc<CrawlControls>,
    metrics: &Metrics,
    filter: UrlFilter,
    router: &ShardRouter,
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

    if task.depth() >= controls.max_depth() || discovered_links.is_empty() {
        return Ok(());
    }

    route_discovered_links(
        router,
        frontier,
        stop_requested,
        controls,
        metrics,
        filter,
        discovered_links,
        task.depth(),
    )
    .await
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
async fn route_discovered_links<const QUEUE: usize, const SEEN: usize>(
    router: &ShardRouter,
    frontier: Arc<Frontier<QUEUE, SEEN>>,
    stop_requested: &AtomicBool,
    controls: Arc<CrawlControls>,
    metrics: &Metrics,
    filter: UrlFilter,
    discovered_links: Vec<Url>,
    parent_depth: u8,
) -> Result<(), TaskError> {
    if discovered_links.is_empty() {
        return Ok(());
    }

    let mut local_links = Vec::new();
    let mut remote: HashMap<usize, Vec<Url>> = HashMap::new();
    for link in discovered_links {
        let owner = router.partition.shard_for_url(&link);
        if owner == router.partition.index() {
            local_links.push(link);
        } else {
            remote.entry(owner).or_default().push(link);
        }
    }

    if !local_links.is_empty() {
        metrics.record_local_shard_links(local_links.len());
        enqueue_discovered_links(
            local_links,
            parent_depth,
            Arc::clone(&frontier),
            stop_requested,
            Arc::clone(&controls),
            metrics,
            Arc::clone(&filter),
        )
        .await?;
    }

    for (shard, links) in remote {
        if let Some(sender) = router.remotes.get(shard) {
            let batch_len = links.len();
            if sender
                .send(LinkBatch {
                    depth: parent_depth,
                    links,
                })
                .await
                .is_err()
            {
                eprintln!(
                    "shard {}: remote shard {shard} channel closed",
                    router.partition.index()
                );
            } else {
                metrics.record_remote_shard_links(batch_len);
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
struct ShardRouter {
    partition: ShardPartition,
    remotes: Arc<Vec<mpsc::Sender<LinkBatch>>>,
}

#[cfg(feature = "multi_thread")]
impl ShardRouter {
    fn new(partition: ShardPartition, remotes: Arc<Vec<mpsc::Sender<LinkBatch>>>) -> Self {
        Self { partition, remotes }
    }
}

#[cfg(feature = "multi_thread")]
async fn shard_inbox_loop<const QUEUE: usize, const SEEN: usize>(
    mut inbox: mpsc::Receiver<LinkBatch>,
    frontier: Arc<Frontier<QUEUE, SEEN>>,
    stop_requested: Arc<AtomicBool>,
    controls: Arc<CrawlControls>,
    metrics: Arc<Metrics>,
    filter: UrlFilter,
) {
    let mut closed = false;
    while let Some(batch) = {
        if stop_requested.load(Ordering::Acquire) && !closed {
            inbox.close();
            closed = true;
        }
        inbox.recv().await
    } {
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
            eprintln!("shard inbox failed for {label}: {message}");
        }
    }
}

#[cfg(not(feature = "multi_thread"))]
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

#[cfg(feature = "multi_thread")]
async fn seed_partitioned_frontier(
    frontier: &Frontier<DEFAULT_FRONTIER_QUEUE, DEFAULT_FRONTIER_SEEN>,
    seeds: &[String],
    partition: &ShardPartition,
) {
    for seed in seeds {
        let owner = partition.shard_for_str(seed);
        if owner != partition.index() {
            continue;
        }
        if let Err(err) = frontier.push_seed_url(seed, 0).await {
            eprintln!(
                "shard {}: failed to enqueue seed {seed}: {err:?}",
                partition.index()
            );
        }
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
    #[cfg(feature = "multi_thread")]
    local_shard_links: AtomicUsize,
    #[cfg(feature = "multi_thread")]
    remote_shard_links: AtomicUsize,
    #[cfg(feature = "multi_thread")]
    remote_batches: AtomicUsize,
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
            TaskErrorKind::Html => {
                self.parse_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[cfg(feature = "multi_thread")]
    fn record_local_shard_links(&self, count: usize) {
        self.local_shard_links.fetch_add(count, Ordering::Relaxed);
    }

    #[cfg(feature = "multi_thread")]
    fn record_remote_shard_links(&self, count: usize) {
        self.remote_shard_links.fetch_add(count, Ordering::Relaxed);
        self.remote_batches.fetch_add(1, Ordering::Relaxed);
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
        #[cfg(feature = "multi_thread")]
        {
            println!(
                "local shard enqueues: {}",
                self.local_shard_links.load(Ordering::Relaxed)
            );
            println!(
                "remote shard links: {} (batches {})",
                self.remote_shard_links.load(Ordering::Relaxed),
                self.remote_batches.load(Ordering::Relaxed)
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskErrorKind {
    Http,
    Parse,
    Frontier,
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
