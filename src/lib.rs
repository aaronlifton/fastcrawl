#![warn(missing_docs)]
//! Core library entry points for the fastcrawl crawler.

pub mod agents;
mod bloom;
pub mod controls;
pub mod embedder;
pub mod embeddings;
pub mod frontier;
pub mod html;
pub mod manifest;
pub mod normalizer;
pub mod runtime;
pub mod vector_store;

pub use agents::{registry, AgentRegistry, CrawlTask, InlineString};
pub use controls::{Cli, CrawlControls};
pub use embeddings::EmbeddedChunkRecord;
pub use frontier::{Frontier, FrontierError, DEFAULT_FRONTIER_QUEUE, DEFAULT_FRONTIER_SEEN};
pub use manifest::ManifestRecord;
pub use normalizer::{
    BlockKind, FetchedPage, NormalizationConfig, NormalizationError, NormalizedChunk,
    NormalizedPage, Normalizer, PageMetadata, SectionHeading, TextBlock,
};
pub use runtime::run as run_crawler;
pub use vector_store::TableName;

#[cfg(feature = "debug_logs")]
#[macro_export]
// This allows use of the `eprintln!` macro via `debug_log!` macro.
macro_rules! debug_log {
        ($($arg:tt)*) => {
            eprintln!($($arg)*);
        };
    }
#[cfg(not(feature = "debug_logs"))]
#[macro_export]
// This effectively disables the `eprintln!` macro, effectively removing it from the code during
// compilation.
macro_rules! debug_log {
    ($($arg:tt)*) => {};
}
