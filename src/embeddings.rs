//! Shared embedding data structures used across pipeline stages.

use serde::{Deserialize, Serialize};

use crate::normalizer::SectionHeading;

/// Output row emitted by embedding jobs and consumed by vector stores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddedChunkRecord {
    /// Source URL.
    pub url: String,
    /// Normalizer-assigned chunk identifier.
    pub chunk_id: usize,
    /// Chunk body text submitted to the embedding model.
    pub text: String,
    /// Hierarchical section breadcrumb for the chunk.
    pub section_path: Vec<SectionHeading>,
    /// Rough token count emitted by the normalizer.
    pub token_estimate: usize,
    /// Model embedding vector.
    pub embedding: Vec<f32>,
    /// Content checksum captured during crawling.
    pub checksum: u32,
    /// Timestamp (ms since epoch) when the page was last fetched.
    pub last_seen_epoch_ms: u64,
}
