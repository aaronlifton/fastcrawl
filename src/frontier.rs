//! Frontier coordination for distributing crawl work to agents.

use crate::agents::MAX_INLINE_URL;
use crate::agents::{AgentRegistry, CrawlTask, InlineString, InlineStringError, SubmitError};
use futures_util::task::AtomicWaker;
use heapless::Deque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::task::yield_now;

/// Default bounded queue depth for the frontier.
pub const DEFAULT_FRONTIER_QUEUE: usize = 2048;
/// Default number of 64-bit words reserved for the Bloom dedupe filter.
pub const DEFAULT_FRONTIER_SEEN: usize = 512;

/// Errors that can emerge while queueing URLs into the frontier.
#[derive(Debug)]
pub enum FrontierError {
    /// The queue is full; the caller retains ownership of the task.
    QueueFull(CrawlTask),
    /// The URL was already scheduled; duplicates are rejected.
    Duplicate(CrawlTask),
    /// The URL exceeded the inline buffer capacity.
    UrlEncoding(InlineStringError),
    /// The frontier is shutting down and no longer accepts work.
    ShuttingDown(CrawlTask),
}

/// A cooperative frontier that deduplicates and dispatches crawl tasks.
pub struct Frontier<const QUEUE: usize, const FILTER_WORDS: usize> {
    queue: Mutex<Deque<CrawlTask, QUEUE>>,
    dedupe: Mutex<BloomFilter<FILTER_WORDS>>,
    pending: AtomicUsize,
    shutdown: AtomicBool,
    waker: AtomicWaker,
}

impl<const QUEUE: usize, const FILTER_WORDS: usize> Frontier<QUEUE, FILTER_WORDS> {
    /// Constructs a new, empty frontier.
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(Deque::new()),
            dedupe: Mutex::new(BloomFilter::new()),
            pending: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            waker: AtomicWaker::new(),
        }
    }

    /// Number of tasks waiting inside the frontier queue.
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::Acquire)
    }

    /// Attempts to insert a raw URL as a new crawl seed.
    pub async fn push_seed_url(&self, url: &str, depth: u8) -> Result<(), FrontierError> {
        let inline = InlineString::<MAX_INLINE_URL>::try_from_str(url)
            .map_err(FrontierError::UrlEncoding)?;
        let task = CrawlTask::new(inline, depth);
        self.push_task(task).await
    }

    /// Attempts to enqueue a crawl task, returning the task on failure.
    pub async fn push_task(&self, task: CrawlTask) -> Result<(), FrontierError> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(FrontierError::ShuttingDown(task));
        }

        let mut dedupe = self.dedupe.lock().await;
        if !dedupe.insert(task.url().as_bytes()) {
            return Err(FrontierError::Duplicate(task));
        }

        self.enqueue_back(task).await
    }

    async fn enqueue_back(&self, task: CrawlTask) -> Result<(), FrontierError> {
        let mut queue = self.queue.lock().await;
        match queue.push_back(task) {
            Ok(()) => {
                self.pending.fetch_add(1, Ordering::Release);
                self.waker.wake();
                Ok(())
            }
            Err(task) => Err(FrontierError::QueueFull(task)),
        }
    }

    async fn enqueue_front(&self, task: CrawlTask) -> Result<(), CrawlTask> {
        let mut queue = self.queue.lock().await;
        match queue.push_front(task) {
            Ok(()) => {
                self.pending.fetch_add(1, Ordering::Release);
                self.waker.wake();
                Ok(())
            }
            Err(task) => Err(task),
        }
    }

    async fn try_next_task(&self) -> Option<CrawlTask> {
        let mut queue = self.queue.lock().await;
        let next = queue.pop_front();
        if next.is_some() {
            self.pending.fetch_sub(1, Ordering::Release);
        }
        next
    }

    fn should_wake(&self) -> bool {
        self.pending.load(Ordering::Acquire) > 0 || self.shutdown.load(Ordering::Acquire)
    }

    /// Blocks until a task is available or shutdown is requested.
    pub async fn next_task(&self) -> Option<CrawlTask> {
        loop {
            if let Some(task) = self.try_next_task().await {
                return Some(task);
            }

            if self.shutdown.load(Ordering::Acquire) {
                return None;
            }

            WaitForFrontier { frontier: self }.await;
        }
    }

    /// Drives the frontier, dispatching tasks into the shared agent registry.
    pub async fn dispatch_loop<const COUNT: usize, const DEPTH: usize>(
        self: Arc<Self>,
        registry: Arc<AgentRegistry<COUNT, DEPTH>>,
    ) {
        while !self.shutdown.load(Ordering::Acquire) {
            match self.next_task().await {
                Some(mut task) => loop {
                    match registry.submit(task).await {
                        Ok(_) => break,
                        Err(SubmitError::QueueFull(returned)) => {
                            task = returned;
                            yield_now().await;
                        }
                        Err(SubmitError::ShuttingDown(returned)) => {
                            let _ = self.enqueue_front(returned).await;
                            self.shutdown();
                            break;
                        }
                    }
                },
                None => break,
            }
        }
    }

    /// Signals shutdown and wakes any waiters.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.waker.wake();
    }
}

struct WaitForFrontier<'a, const QUEUE: usize, const FILTER_WORDS: usize> {
    frontier: &'a Frontier<QUEUE, FILTER_WORDS>,
}

impl<'a, const QUEUE: usize, const FILTER_WORDS: usize> core::future::Future
    for WaitForFrontier<'a, QUEUE, FILTER_WORDS>
{
    type Output = ();

    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<Self::Output> {
        if self.frontier.should_wake() {
            core::task::Poll::Ready(())
        } else {
            self.frontier.waker.register(cx.waker());
            if self.frontier.should_wake() {
                core::task::Poll::Ready(())
            } else {
                core::task::Poll::Pending
            }
        }
    }
}

struct BloomFilter<const WORDS: usize> {
    words: [u64; WORDS],
}

impl<const WORDS: usize> BloomFilter<WORDS> {
    const fn new() -> Self {
        Self {
            words: [0u64; WORDS],
        }
    }

    fn insert(&mut self, data: &[u8]) -> bool {
        if WORDS == 0 {
            return false;
        }
        let bit_count = WORDS * 64;
        let mut inserted = false;
        for &seed in BLOOM_HASH_SEEDS.iter() {
            let hash = bloom_hash(data, seed);
            let idx = (hash as usize) % bit_count;
            let word = idx / 64;
            let bit = idx % 64;
            let mask = 1u64 << bit;
            if self.words[word] & mask == 0 {
                inserted = true;
                self.words[word] |= mask;
            }
        }
        inserted
    }
}

const BLOOM_HASH_SEEDS: [u64; 3] = [0x517cc1b727220a95, 0x6d0f27bdceb7b067, 0x9e3779b185ebca87];

fn bloom_hash(data: &[u8], seed: u64) -> u64 {
    let mut hash = seed ^ data.len() as u64;
    for &byte in data {
        hash ^= (byte as u64).wrapping_mul(0x100000001b3);
        hash = hash.rotate_left(13).wrapping_mul(0xff51afd7ed558ccd);
    }
    hash ^ (hash >> 33)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::AgentRegistry;
    use std::sync::Arc;

    #[tokio::test(flavor = "current_thread")]
    async fn seeds_flow_into_agent_queue() {
        let registry = Arc::new(AgentRegistry::<1, 4>::new());
        let frontier = Arc::new(Frontier::<4, 4>::new());

        let dispatcher = {
            let frontier = Arc::clone(&frontier);
            let registry = Arc::clone(&registry);
            tokio::spawn(async move {
                frontier.dispatch_loop(registry).await;
            })
        };

        frontier
            .push_seed_url("https://seed.test", 0)
            .await
            .expect("seed accepted");

        let agent = registry.agent(0).expect("agent exists");
        let task = agent.next_task().await.expect("frontier dispatched task");
        assert_eq!(task.url(), "https://seed.test");

        frontier.shutdown();
        dispatcher.await.expect("dispatcher joined");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn duplicate_urls_rejected() {
        let frontier = Frontier::<2, 2>::new();

        frontier
            .push_seed_url("https://dup.test", 0)
            .await
            .expect("first seed");

        match frontier
            .push_seed_url("https://dup.test", 1)
            .await
            .expect_err("duplicate rejected")
        {
            FrontierError::Duplicate(task) => {
                assert_eq!(task.depth(), 1);
            }
            other => panic!("expected duplicate error, got {other:?}"),
        }
    }
}
