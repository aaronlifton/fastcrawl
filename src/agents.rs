//! Lock-free leaning agent coordination with minimal heap overhead.

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use futures_util::task::AtomicWaker;
use heapless::Deque;
use std::array::from_fn;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Duration;
use tokio::sync::Mutex;

#[cfg(not(target_has_atomic = "32"))]
compile_error!("agents module requires target_has_atomic = \"32\" for atomic wait/notify support");

/// Default number of agents in the global registry.
pub const DEFAULT_AGENT_CAPACITY: usize = 8;
/// Per-agent bounded queue depth.
pub const DEFAULT_QUEUE_DEPTH: usize = 64;
/// Inline byte capacity reserved for URLs, avoiding `String` allocations.
pub const MAX_INLINE_URL: usize = 256;

static GLOBAL_REGISTRY: OnceLock<AgentRegistry<DEFAULT_AGENT_CAPACITY, DEFAULT_QUEUE_DEPTH>> =
    OnceLock::new();

/// Returns the lazily initialized global agent registry.
pub fn registry() -> &'static AgentRegistry<DEFAULT_AGENT_CAPACITY, DEFAULT_QUEUE_DEPTH> {
    GLOBAL_REGISTRY.get_or_init(AgentRegistry::new)
}

/// Error emitted when an inline string cannot be constructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InlineStringError {
    /// Input did not fit in the fixed-capacity buffer.
    TooLong {
        /// Number of bytes available in the inline buffer.
        capacity: usize,
        /// Bytes the caller attempted to copy.
        attempted: usize,
    },
}

/// An inline, stack-allocated UTF-8 string with fixed capacity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InlineString<const N: usize> {
    len: u16,
    buf: [u8; N],
}

impl<const N: usize> InlineString<N> {
    /// Attempts to copy the provided string into the inline buffer.
    pub fn try_from_str(input: &str) -> Result<Self, InlineStringError> {
        let bytes = input.as_bytes();
        if bytes.len() > N {
            return Err(InlineStringError::TooLong {
                capacity: N,
                attempted: bytes.len(),
            });
        }

        let mut buf = [0u8; N];
        buf[..bytes.len()].copy_from_slice(bytes);
        Ok(Self {
            len: bytes.len() as u16,
            buf,
        })
    }

    /// Returns the stored string slice.
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.buf[..self.len as usize]).expect("inline utf8 validated on write")
    }
}

/// Description of a crawl task that can run without heap cloning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrawlTask {
    url: InlineString<MAX_INLINE_URL>,
    depth: u8,
    politeness_delay_ms: u32,
}

impl CrawlTask {
    /// Creates a new crawl task.
    pub fn new(url: InlineString<MAX_INLINE_URL>, depth: u8) -> Self {
        Self {
            url,
            depth,
            politeness_delay_ms: 0,
        }
    }

    /// Adds an optional politeness delay budget.
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.politeness_delay_ms = delay.as_millis().min(u32::MAX as u128) as u32;
        self
    }

    /// Returns the crawl depth.
    pub fn depth(&self) -> u8 {
        self.depth
    }

    /// Returns the politeness delay in milliseconds.
    pub fn politeness_delay_ms(&self) -> u32 {
        self.politeness_delay_ms
    }

    /// Returns the target URL.
    pub fn url(&self) -> &str {
        self.url.as_str()
    }
}

/// Errors that can emerge while submitting work into the agent queue.
#[derive(Debug)]
pub enum SubmitError {
    /// The agent is shutting down and refuses more work.
    ShuttingDown(CrawlTask),
    /// The bounded queue is at capacity.
    QueueFull(CrawlTask),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubmitErrorKind {
    QueueFull,
    ShuttingDown,
}

impl SubmitError {
    /// Recover the original task payload.
    pub fn into_task(self) -> CrawlTask {
        match self {
            SubmitError::ShuttingDown(task) | SubmitError::QueueFull(task) => task,
        }
    }

    fn kind(&self) -> SubmitErrorKind {
        match self {
            SubmitError::QueueFull(_) => SubmitErrorKind::QueueFull,
            SubmitError::ShuttingDown(_) => SubmitErrorKind::ShuttingDown,
        }
    }
}

/// Pool-wide coordination for crawl agents.
pub struct AgentRegistry<const COUNT: usize, const DEPTH: usize> {
    agents: [Agent<DEPTH>; COUNT],
    rr_cursor: AtomicUsize,
}

impl<const COUNT: usize, const DEPTH: usize> AgentRegistry<COUNT, DEPTH> {
    /// Builds a new registry with COUNT agents.
    pub fn new() -> Self {
        Self {
            agents: from_fn(Agent::new),
            rr_cursor: AtomicUsize::new(0),
        }
    }

    /// Returns an iterator over all registered agents.
    pub fn iter(&self) -> impl Iterator<Item = &Agent<DEPTH>> {
        self.agents.iter()
    }

    /// Pulls a handle for a specific agent id.
    pub fn agent(&self, id: usize) -> Option<&Agent<DEPTH>> {
        self.agents.get(id)
    }

    /// Submits a task in round-robin order, minimizing heap churn.
    pub async fn submit(&self, mut task: CrawlTask) -> Result<usize, SubmitError> {
        let start = self.rr_cursor.fetch_add(1, Ordering::AcqRel);
        let mut last_kind = SubmitErrorKind::QueueFull;
        for offset in 0..COUNT {
            let idx = (start + offset) % COUNT;
            match self.agents[idx].submit(task).await {
                Ok(()) => return Ok(idx),
                Err(err) => {
                    last_kind = err.kind();
                    task = err.into_task();
                }
            }
        }

        Err(match last_kind {
            SubmitErrorKind::QueueFull => SubmitError::QueueFull(task),
            SubmitErrorKind::ShuttingDown => SubmitError::ShuttingDown(task),
        })
    }
}

impl<const COUNT: usize, const DEPTH: usize> Default for AgentRegistry<COUNT, DEPTH> {
    fn default() -> Self {
        Self::new()
    }
}

/// A single crawling agent with a bounded queue and atomic wakeups.
pub struct Agent<const DEPTH: usize> {
    id: usize,
    queue: Mutex<Deque<CrawlTask, DEPTH>>,
    pending: AtomicUsize,
    shutdown: AtomicBool,
    waker: AtomicWaker,
}

impl<const DEPTH: usize> Agent<DEPTH> {
    fn new(id: usize) -> Self {
        Self {
            id,
            queue: Mutex::new(Deque::new()),
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            waker: AtomicWaker::new(),
        }
    }

    /// Agent identifier inside the registry.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Number of tasks waiting to be processed.
    pub fn backlog(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }

    /// Attempt to enqueue a task without heap allocation.
    pub async fn submit(&self, task: CrawlTask) -> Result<(), SubmitError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(SubmitError::ShuttingDown(task));
        }

        let mut queue = self.queue.lock().await;
        if self.shutdown.load(Ordering::Acquire) {
            return Err(SubmitError::ShuttingDown(task));
        }

        match queue.push_back(task) {
            Ok(()) => {
                self.pending.fetch_add(1, Ordering::Release);
                self.waker.wake();
                Ok(())
            }
            Err(task) => Err(SubmitError::QueueFull(task)),
        }
    }

    /// Try to fetch a task immediately.
    pub async fn try_next_task(&self) -> Option<CrawlTask> {
        let mut queue = self.queue.lock().await;
        let next = queue.pop_front();
        if next.is_some() {
            self.pending.fetch_sub(1, Ordering::Release);
        }
        next
    }

    /// Awaits until a task becomes available or the agent shuts down.
    pub async fn next_task(&self) -> Option<CrawlTask> {
        loop {
            if let Some(task) = self.try_next_task().await {
                return Some(task);
            }

            if self.shutdown.load(Ordering::Acquire) {
                return None;
            }

            WaitForWake { agent: self }.await;
        }
    }

    /// Signals the agent to stop accepting work and wakes all waiters.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.waker.wake();
    }

    fn should_wake(&self) -> bool {
        self.pending.load(Ordering::Acquire) > 0 || self.shutdown.load(Ordering::Acquire)
    }
}

struct WaitForWake<'a, const DEPTH: usize> {
    agent: &'a Agent<DEPTH>,
}

impl<'a, const DEPTH: usize> Future for WaitForWake<'a, DEPTH> {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.agent.should_wake() {
            Poll::Ready(())
        } else {
            self.agent.waker.register(cx.waker());
            if self.agent.should_wake() {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn inline_string_round_trip() {
        let inline = InlineString::<16>::try_from_str("https://r").expect("fits");
        assert_eq!(inline.as_str(), "https://r");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn agent_round_robin_submission() {
        let registry: AgentRegistry<2, 4> = AgentRegistry::new();
        let url = InlineString::<MAX_INLINE_URL>::try_from_str("https://example.com").unwrap();
        let task = CrawlTask::new(url, 0);
        let idx = registry.submit(task).await.expect("queue accepts");
        assert!(idx < 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn async_workers_wake() {
        let agent: Arc<Agent<4>> = Arc::new(Agent::new(0));
        let worker = {
            let agent = Arc::clone(&agent);
            tokio::spawn(async move { agent.next_task().await })
        };

        let url = InlineString::<MAX_INLINE_URL>::try_from_str("https://wake.test").unwrap();
        let task = CrawlTask::new(url, 0);
        agent.submit(task).await.unwrap();

        let result = worker.await.unwrap();
        assert_eq!(result.unwrap().url(), "https://wake.test");
    }
}
