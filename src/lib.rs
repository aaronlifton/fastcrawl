#![warn(missing_docs)]
//! Core library entry points for the fastcrawl crawler.

pub mod agents;
pub mod frontier;

pub use agents::{registry, AgentRegistry, CrawlTask, InlineString};
pub use frontier::{Frontier, FrontierError, DEFAULT_FRONTIER_QUEUE, DEFAULT_FRONTIER_SEEN};
