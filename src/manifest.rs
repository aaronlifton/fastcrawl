//! Shared manifest records emitted during normalization for incremental refresh pipelines.

use crate::normalizer::NormalizedPage;
use serde::{Deserialize, Serialize};

/// Digest entry describing the latest snapshot of a URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestRecord {
    /// Canonical page URL.
    pub url: String,
    /// Raw body checksum (CRC32).
    pub checksum: u32,
    /// Epoch milliseconds when the page was last seen.
    pub last_seen_epoch_ms: u64,
    /// Whether the checksum differs from the prior observation.
    pub changed: bool,
}

impl ManifestRecord {
    /// Creates a manifest record from the provided components.
    pub fn new(url: String, checksum: u32, last_seen_epoch_ms: u64, changed: bool) -> Self {
        Self {
            url,
            checksum,
            last_seen_epoch_ms,
            changed,
        }
    }

    /// Helper that derives a manifest record from a normalized page plus a `changed` flag.
    pub fn from_page(page: &NormalizedPage, changed: bool) -> Self {
        let meta = &page.metadata;
        Self::new(
            meta.url.to_string(),
            meta.checksum,
            meta.fetched_epoch_ms(),
            changed,
        )
    }
}
