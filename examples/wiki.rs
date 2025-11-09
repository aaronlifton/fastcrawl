use fastcrawl::agents::{
    AgentRegistry, CrawlTask, InlineString, DEFAULT_AGENT_CAPACITY, DEFAULT_QUEUE_DEPTH,
};
use fastcrawl::frontier::{Frontier, FrontierError, DEFAULT_FRONTIER_QUEUE, DEFAULT_FRONTIER_SEEN};
use futures_util::future::join_all;
use reqwest::Client;
use scraper::{Html, Selector};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use url::Url;

const MAX_DEPTH: u8 = 4;
const MAX_LINKS_PER_PAGE: usize = 16;
const RUN_DURATION: Duration = Duration::from_secs(30);
const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(100);
const USER_AGENT: &str = "fastcrawl-example/0.1 (+https://github.com/aaronlifton/fastcrawl)";

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("booting fastcrawl wikipedia demo…");

    let registry = Arc::new(AgentRegistry::<DEFAULT_AGENT_CAPACITY, DEFAULT_QUEUE_DEPTH>::new());
    let frontier = Arc::new(Frontier::<DEFAULT_FRONTIER_QUEUE, DEFAULT_FRONTIER_SEEN>::new());
    let stop_requested = Arc::new(AtomicBool::new(false));
    let active_tasks = Arc::new(AtomicUsize::new(0));
    let metrics = Arc::new(Metrics::default());
    let start = Instant::now();

    let client = Client::builder()
        .user_agent(USER_AGENT)
        .redirect(reqwest::redirect::Policy::limited(5))
        .timeout(Duration::from_secs(10))
        .build()?;

    for seed in [
        "https://en.wikipedia.org/wiki/Web_crawler",
        "https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol",
        "https://en.wikipedia.org/wiki/Capybara",
        "https://en.wikipedia.org/wiki/Cat",
    ] {
        frontier
            .push_seed_url(seed, 0)
            .await
            .expect("seed enqueue succeeds");
    }

    let dispatcher = {
        let frontier = Arc::clone(&frontier);
        let registry = Arc::clone(&registry);
        tokio::spawn(async move {
            frontier.dispatch_loop(registry).await;
        })
    };

    let mut workers = Vec::new();
    for id in 0..DEFAULT_AGENT_CAPACITY {
        let registry = Arc::clone(&registry);
        let frontier = Arc::clone(&frontier);
        let client = client.clone();
        let stop_requested = Arc::clone(&stop_requested);
        let active_tasks = Arc::clone(&active_tasks);
        let metrics = Arc::clone(&metrics);
        workers.push(tokio::spawn(async move {
            worker_loop(
                id,
                registry,
                frontier,
                client,
                stop_requested,
                active_tasks,
                metrics,
            )
            .await;
        }));
    }

    sleep(RUN_DURATION).await;
    stop_requested.store(true, Ordering::Release);
    wait_for_drain(registry.as_ref(), frontier.as_ref(), active_tasks.as_ref()).await;
    frontier.shutdown();
    for agent in registry.iter() {
        agent.shutdown();
    }

    dispatcher
        .await
        .expect("frontier dispatcher finished cleanly");
    join_all(workers).await;

    metrics.report(start.elapsed());

    println!("crawl finished.");
    Ok(())
}

async fn worker_loop<
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
    metrics: Arc<Metrics>,
) {
    let Some(agent) = registry.agent(agent_id) else {
        eprintln!("worker {agent_id}: agent missing");
        return;
    };

    while let Some(task) = agent.next_task().await {
        let _active = ActiveTaskGuard::new(active_tasks.as_ref());
        if let Err(err) = handle_task(
            &client,
            Arc::clone(&frontier),
            task,
            stop_requested.as_ref(),
            metrics.as_ref(),
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

struct ActiveTaskGuard<'a> {
    counter: &'a AtomicUsize,
}

impl<'a> ActiveTaskGuard<'a> {
    fn new(counter: &'a AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self { counter }
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
        }
    }

    fn report(&self, elapsed: Duration) {
        let secs = elapsed.as_secs_f32();
        println!("--- crawl metrics ({secs:.2}s) ---");
        println!(
            "pages fetched: {}",
            self.pages_fetched.load(Ordering::Relaxed)
        );
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

impl Drop for ActiveTaskGuard<'_> {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TaskErrorKind {
    Http,
    Parse,
    Frontier,
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
}

async fn handle_task<const QUEUE: usize, const SEEN: usize>(
    client: &Client,
    frontier: Arc<Frontier<QUEUE, SEEN>>,
    task: CrawlTask,
    stop_requested: &AtomicBool,
    metrics: &Metrics,
) -> Result<(), TaskError> {
    let url = task.url().to_string();
    println!("[depth {}] {}", task.depth(), url);

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|err| TaskError::http(&url, err))?;

    if !response.status().is_success() {
        return Ok(());
    }

    let body = response
        .text()
        .await
        .map_err(|err| TaskError::http(&url, err))?;

    metrics.record_page_fetched();

    if task.depth() >= MAX_DEPTH {
        return Ok(());
    }

    let base = Url::parse(&url).map_err(|err| TaskError::parse(&url, err))?;
    for discovered in extract_links(&base, &body) {
        if stop_requested.load(Ordering::Acquire) {
            break;
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
        let next = CrawlTask::new(inline, task.depth() + 1).with_delay(Duration::from_millis(250));

        match frontier.push_task(next).await {
            Ok(()) => metrics.record_url_enqueued(),
            Err(FrontierError::Duplicate(_)) => metrics.record_duplicate(),
            Err(err) => {
                return Err(TaskError::frontier(discovered.as_str(), err));
            }
        }
    }

    Ok(())
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

const ARTICLE_PREFIX: &str = "/wiki/";
const BLOCKED_NAMESPACES: &[&str] = &[
    "Special:",
    "Help:",
    "Help_talk:",
    "Wikipedia:",
    "Talk:",
    "User:",
    "User_talk:",
    "File:",
    "Template:",
    "Portal:",
    "Portal_talk:",
    "Category:",
    "Book:",
    "Draft:",
];
const BLOCKED_QUERY_SNIPPETS: &[&str] = &[
    "action=",
    "feed=",
    "printable=yes",
    "returnto=",
    "useskin=",
    "oldid=",
    "diff=",
    "curid=",
    "Special:",
];

fn extract_links(base: &Url, body: &str) -> Vec<Url> {
    let document = Html::parse_document(body);
    let selector = Selector::parse("a[href]").expect("valid selector");

    document
        .select(&selector)
        .filter_map(|element| element.value().attr("href"))
        .filter_map(|href| base.join(href).ok())
        .filter(|url| is_allowed_article(url))
        .filter(|url| !query_has_blocked_params(url))
        .take(MAX_LINKS_PER_PAGE)
        .collect()
}

fn is_allowed_article(url: &Url) -> bool {
    if url.scheme() != "https" || url.domain() != Some("en.wikipedia.org") {
        return false;
    }

    if url.fragment().is_some() {
        return false;
    }

    let path = url.path();
    let Some(article) = path.strip_prefix(ARTICLE_PREFIX) else {
        return false;
    };

    !BLOCKED_NAMESPACES
        .iter()
        .any(|prefix| article.starts_with(prefix))
}

fn query_has_blocked_params(url: &Url) -> bool {
    url.query()
        .map(|query| {
            BLOCKED_QUERY_SNIPPETS
                .iter()
                .any(|blocked| query.contains(blocked))
        })
        .unwrap_or(false)
}
