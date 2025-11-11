#![warn(missing_docs)]
//! Core library entry points for the fastcrawl crawler.

pub mod agents;
mod bloom;
pub mod controls;
pub mod frontier;
pub mod html;
pub mod runtime;

pub use agents::{registry, AgentRegistry, CrawlTask, InlineString};
pub use controls::{Cli, CrawlControls};
pub use frontier::{Frontier, FrontierError, DEFAULT_FRONTIER_QUEUE, DEFAULT_FRONTIER_SEEN};
pub use runtime::run as run_crawler;
