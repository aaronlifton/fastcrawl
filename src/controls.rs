//! Crawl throttle and filtering controls shared across executors.

use crate::normalizer::NormalizationConfig;
use clap::Parser;
#[cfg(feature = "multi_thread")]
use clap::ValueEnum;
use std::path::PathBuf;
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

    /// Enable corpus normalization pipeline and JSONL output
    #[arg(long, env = "FASTCRAWL_NORMALIZE", default_value_t = false)]
    pub normalize: bool,

    /// Output path for normalized JSONL batches (overwrites existing file)
    #[arg(
        long,
        env = "FASTCRAWL_NORMALIZE_JSONL",
        default_value = "normalized_pages.jsonl"
    )]
    pub normalize_jsonl: String,

    /// Optional manifest JSONL capturing per-URL digests (checksum + last seen)
    #[arg(long, env = "FASTCRAWL_NORMALIZE_MANIFEST")]
    pub normalize_manifest_jsonl: Option<PathBuf>,

    /// Target tokens per chunk emitted by the normalizer
    #[arg(long, env = "FASTCRAWL_NORMALIZE_TOKENS", default_value_t = 256)]
    pub normalize_chunk_tokens: usize,

    /// Token overlap between neighboring chunks
    #[arg(long, env = "FASTCRAWL_NORMALIZE_OVERLAP", default_value_t = 48)]
    pub normalize_overlap_tokens: usize,

    /// Maximum text blocks to keep before truncating normalization
    #[arg(long, env = "FASTCRAWL_NORMALIZE_MAX_BLOCKS", default_value_t = 8192)]
    pub normalize_max_blocks: usize,

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

    /// Emit per-shard channel-closed logs (helps debugging shutdown races).
    #[cfg(feature = "multi_thread")]
    #[arg(long, env = "FASTCRAWL_REMOTE_CHANNEL_LOGS", default_value_t = false)]
    pub remote_channel_logs: bool,
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

    /// Returns normalization settings when enabled.
    pub fn normalization_settings(&self) -> Option<NormalizationSettings> {
        if !self.normalize {
            return None;
        }

        let chunk_target = self.normalize_chunk_tokens.max(1);
        let chunk_overlap = self
            .normalize_overlap_tokens
            .min(chunk_target.saturating_sub(1));
        let max_blocks = self.normalize_max_blocks.max(1);

        Some(NormalizationSettings {
            output_path: PathBuf::from(&self.normalize_jsonl),
            manifest_path: self.normalize_manifest_jsonl.clone(),
            config: NormalizationConfig {
                chunk_target_tokens: chunk_target,
                chunk_overlap_tokens: chunk_overlap,
                max_blocks,
            },
        })
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
            remote_channel_logs: self.remote_channel_logs,
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
    /// Whether to log channel-closed warnings for cross-shard sends.
    pub remote_channel_logs: bool,
}

/// Settings controlling corpus normalization outputs.
#[derive(Debug, Clone)]
pub struct NormalizationSettings {
    /// Filesystem path that will receive newline-delimited JSON.
    pub output_path: PathBuf,
    /// Optional path for digest manifest records.
    pub manifest_path: Option<PathBuf>,
    /// Chunking/cleanup configuration applied to each page.
    pub config: NormalizationConfig,
}
