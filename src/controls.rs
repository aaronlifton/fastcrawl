//! Crawl throttle and filtering controls shared across executors.

use clap::Parser;
#[cfg(feature = "multi_thread")]
use clap::ValueEnum;
use std::time::Duration;

/// Tunable knobs that bound crawl behavior.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CrawlControls {
    max_depth: u8,
    max_links_per_page: usize,
    politeness_delay: Duration,
    allowed_domains: Vec<String>,
}

impl CrawlControls {
    /// Constructs a new set of crawl controls.
    pub fn new(
        max_depth: u8,
        max_links_per_page: usize,
        politeness_delay: Duration,
        allowed_domains: Vec<String>,
    ) -> Self {
        Self {
            max_depth,
            max_links_per_page,
            politeness_delay,
            allowed_domains,
        }
    }

    /// Maximum crawl depth allowed.
    pub fn max_depth(&self) -> u8 {
        self.max_depth
    }

    /// Maximum number of links extracted per page.
    pub fn max_links_per_page(&self) -> usize {
        self.max_links_per_page
    }

    /// Time to wait between fetching successive links.
    pub fn politeness_delay(&self) -> Duration {
        self.politeness_delay
    }

    /// Returns the allowlist of domains.
    pub fn allowed_domains(&self) -> &[String] {
        &self.allowed_domains
    }

    /// Determines whether the provided domain passes the allowlist.
    pub fn is_domain_allowed(&self, domain: &str) -> bool {
        self.allowed_domains.iter().any(|allowed| allowed == domain)
    }
}

impl Default for CrawlControls {
    fn default() -> Self {
        Self {
            max_depth: 4,
            max_links_per_page: 16,
            politeness_delay: Duration::from_millis(250),
            allowed_domains: Vec::new(),
        }
    }
}

/// Command-line interface shared by binaries that want crawl controls.
#[derive(Parser, Debug, Clone)]
#[command(name = "fastcrawl", about = "Configurable crawler controls")]
pub struct Cli {
    /// Seconds to run before requesting shutdown
    #[arg(long, env = "FASTCRAWL_DURATION", default_value_t = 60)]
    pub duration_secs: u64,

    /// Maximum crawl depth
    #[arg(long, env = "FASTCRAWL_MAX_DEPTH", default_value_t = 4)]
    pub max_depth: u8,

    /// Maximum links captured per page
    #[arg(long, env = "FASTCRAWL_MAX_LINKS", default_value_t = 16)]
    pub max_links_per_page: usize,

    /// Milliseconds to wait before following discovered links
    #[arg(long, env = "FASTCRAWL_POLITENESS_MS", default_value_t = 250)]
    pub politeness_ms: u64,

    /// Domain allowlist, comma separated
    #[arg(long, env = "FASTCRAWL_DOMAINS", default_value = "en.wikipedia.org")]
    pub allowed_domains: String,

    /// Shard partitioning strategy (multi-thread feature only)
    #[cfg(feature = "multi_thread")]
    #[arg(long, env = "FASTCRAWL_PARTITION", default_value = "hash")]
    pub partition: PartitionStrategyArg,

    /// Number of wiki prefix buckets (0 = auto = shard count)
    #[cfg(feature = "multi_thread")]
    #[arg(long, env = "FASTCRAWL_PARTITION_BUCKETS", default_value_t = 0)]
    pub partition_buckets: usize,

    /// Treat namespaces (e.g., Talk:, Help:) as part of the partition key
    #[cfg(feature = "multi_thread")]
    #[arg(long, env = "FASTCRAWL_PARTITION_NAMESPACE", default_value_t = false)]
    pub partition_namespace: bool,

    /// Maximum remote links to buffer before flushing to another shard (0 = default)
    #[cfg(feature = "multi_thread")]
    #[arg(long, env = "FASTCRAWL_REMOTE_BATCH_SIZE", default_value_t = 0)]
    pub remote_batch_size: usize,
}

impl Cli {
    /// Converts the parsed CLI into `CrawlControls`.
    pub fn build_controls(&self) -> CrawlControls {
        CrawlControls::new(
            self.max_depth,
            self.max_links_per_page,
            Duration::from_millis(self.politeness_ms),
            self.domains_vec(),
        )
    }

    /// Returns the requested run duration.
    pub fn run_duration(&self) -> Duration {
        Duration::from_secs(self.duration_secs)
    }

    fn domains_vec(&self) -> Vec<String> {
        self.allowed_domains
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    #[cfg(feature = "multi_thread")]
    /// Returns the requested sharding strategy when multi-threading is enabled.
    pub fn partition_settings(&self) -> PartitionSettings {
        PartitionSettings {
            strategy: self.partition,
            wiki_bucket_count: (self.partition_buckets > 0).then_some(self.partition_buckets),
            wiki_include_namespace: self.partition_namespace,
            remote_batch_size: (self.remote_batch_size > 0).then_some(self.remote_batch_size),
        }
    }
}

#[cfg(feature = "multi_thread")]
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
/// Strategies for assigning discovered URLs to shards in multi-thread mode.
pub enum PartitionStrategyArg {
    /// Hash URLs evenly across shards (default).
    Hash,
    /// Use Wikipedia-style namespace/title prefixes to keep related pages together.
    WikiPrefix,
}

#[cfg(feature = "multi_thread")]
#[derive(Copy, Clone, Debug)]
/// Parsed partition configuration used by the runtime.
pub struct PartitionSettings {
    /// Selected strategy variant.
    pub strategy: PartitionStrategyArg,
    /// Optional bucket count override for wiki prefix strategy (defaults to shard count when `None`).
    pub wiki_bucket_count: Option<usize>,
    /// Whether to incorporate namespace prefixes (e.g., `Talk:`) into the partition key.
    pub wiki_include_namespace: bool,
    /// Optional remote batch size override.
    pub remote_batch_size: Option<usize>,
}
