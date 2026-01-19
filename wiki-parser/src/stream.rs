//! Streaming block extraction using `lol_html`.

use crate::{
    debug_log, BlockKind, ContentBlock, IGNORE_CLASSES, IGNORE_CLASS_PREFIXES, IGNORE_TAGS,
};
use html_escape::decode_html_entities;
use lol_html::html_content::Element;
use lol_html::{element, text, HtmlRewriter, OutputSink, Settings};
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};

/// Streaming block extractor for Wikipedia HTML.
pub struct StreamingExtractor {
    rewriter: HtmlRewriter<'static, NoopSink>,
    state: Arc<Mutex<StreamState>>,
}

impl StreamingExtractor {
    /// Creates a new streaming extractor.
    pub fn new() -> Self {
        let state = Arc::new(Mutex::new(StreamState::new()));
        let element_state = Arc::clone(&state);
        let text_state = Arc::clone(&state);

        let element_handler = element!("*", move |el: &mut Element<'_, '_>| {
            let tag = el.tag_name().to_ascii_lowercase();
            let id_attr = el.get_attribute("id");
            let class_attr = el.get_attribute("class");

            let mut has_parser_output = false;
            let mut has_ignore_class = false;
            if let Some(classes) = class_attr.as_ref() {
                for class in classes.split_whitespace() {
                    if class == "mw-parser-output" {
                        has_parser_output = true;
                    }
                    if should_ignore_class(class) {
                        has_ignore_class = true;
                    }
                }
            }

            let is_body = tag == "body";
            let is_mw_content = id_attr.as_deref() == Some("mw-content-text");
            let is_body_content = id_attr.as_deref() == Some("bodyContent");

            let mut state = element_state
                .lock()
                .unwrap_or_else(|_| panic!("streaming parser state mutex poisoned"));
            state.depth = state.depth.saturating_add(1);
            let depth = state.depth;

            if is_body {
                state.body_depth = Some(depth);
                maybe_set_root(&mut state, ROOT_BODY, depth);
            }
            if is_mw_content {
                state.mw_content_depth = Some(depth);
                maybe_set_root(&mut state, ROOT_MW_CONTENT, depth);
            }
            if is_body_content {
                state.body_content_depth = Some(depth);
                maybe_set_root(&mut state, ROOT_BODY_CONTENT, depth);
            }
            if has_parser_output {
                let priority = if state.mw_content_depth.is_some() {
                    ROOT_MW_PARSER
                } else if state.body_content_depth.is_some() {
                    ROOT_BODY_PARSER
                } else {
                    ROOT_BODY_PARSER_FALLBACK
                };
                maybe_set_root(&mut state, priority, depth);
            }

            let ignore_tag = should_ignore_tag(&tag);
            let mut ignored = ignore_tag || has_ignore_class;
            if tag == "body" || tag == "html" {
                ignored = false;
            }
            if ignored {
                if ignored {
                    debug_log!(
                        "ignored tag={} class={:?} ignore_tag={} has_ignore_class={}",
                        tag,
                        class_attr,
                        ignore_tag,
                        has_ignore_class
                    );
                }
                state.ignore_depth = state.ignore_depth.saturating_add(1);
            }
            eprintln!("tag={} id={} class={}", tag, id_attr.take(), class_attr);

            // debug_log!(
            //     "tag={} in_root={} root_depth={:?} root_closed={} depth={}
            // ignore_depth={}",
            //     tag,
            //     state.in_root(),
            //     state.root_depth,
            //     state.root_closed,
            //     state.depth,
            //     state.ignore_depth
            // );
            let started_block = if !ignored && state.in_root() && is_block_tag(&tag) {
                debug_log!("block candidate tag={}", el.tag_name());
                let kind = super::classify_block(&tag);
                let level = super::heading_level(&tag);
                state.stack.push(ActiveBlock {
                    kind,
                    level,
                    text: String::new(),
                });
                true
            } else {
                false
            };

            let element_state = Arc::clone(&element_state);
            if let Some(handlers) = el.end_tag_handlers() {
                handlers.push(Box::new(move |end| {
                    let _ = end;
                    let mut state = element_state
                        .lock()
                        .unwrap_or_else(|_| panic!("streaming parser state mutex poisoned"));
                    if ignored {
                        state.ignore_depth = state.ignore_depth.saturating_sub(1);
                    }

                    if started_block {
                        if let Some(block) = state.stack.pop() {
                            if let Some(block) = finalize_block(&mut state, block) {
                                debug_log!("block pushed: {:?}", block.kind);
                                state.blocks.push(block);
                            }
                        }
                    }

                    if is_mw_content && state.mw_content_depth == Some(depth) {
                        state.mw_content_depth = None;
                    }
                    if is_body_content && state.body_content_depth == Some(depth) {
                        state.body_content_depth = None;
                    }
                    if is_body && state.body_depth == Some(depth) {
                        state.body_depth = None;
                    }

                    // debug_log!("root_depth={:?}", state.root_depth);
                    if state.root_depth == Some(depth) {
                        state.root_depth = None;
                        state.root_closed = true;
                        debug_log!("root closed. root_depth set to {:?}", state.root_depth);
                    }

                    state.depth = state.depth.saturating_sub(1);
                    Ok(())
                }));
            }
            Ok(())
        });

        let text_handler = text!("*", move |chunk| {
            let mut state = text_state
                .lock()
                .unwrap_or_else(|_| panic!("streaming parser state mutex poisoned"));
            if state.ignore_depth > 0 || !state.in_root() {
                return Ok(());
            }
            let fragment = chunk.as_str();
            if !fragment.is_empty() {
                state.fallback.push_str(fragment);
                if let Some(active) = state.stack.last_mut() {
                    active.text.push_str(fragment);
                }
            }
            Ok(())
        });

        let rewriter = HtmlRewriter::new(
            Settings {
                element_content_handlers: vec![element_handler, text_handler],
                ..Settings::default()
            },
            NoopSink,
        );

        Self { rewriter, state }
    }

    /// Streams a chunk of HTML into the extractor.
    pub fn write(&mut self, chunk: &[u8]) -> Result<(), StreamError> {
        self.rewriter.write(chunk).map_err(StreamError::Rewrite)
    }

    /// Finalizes the stream and returns collected blocks.
    pub fn finish(self) -> Result<Vec<ContentBlock>, StreamError> {
        let StreamingExtractor { rewriter, state } = self;
        rewriter.end().map_err(StreamError::Rewrite)?;

        let state = Arc::try_unwrap(state).map_err(|_| StreamError::CollectorInUse)?;
        let mut state = state
            .into_inner()
            .map_err(|_| StreamError::CollectorPoisoned)?;
        if !state.stack.is_empty() {
            let drained: Vec<_> = state.stack.drain(..).collect();
            for block in drained {
                if let Some(block) = finalize_block(&mut state, block) {
                    debug_log!("block pushed: {:?}", block.kind);
                    state.blocks.push(block);
                }
            }
        }

        if state.blocks.is_empty() {
            let normalized = super::normalize_whitespace(&decode_entities(&state.fallback));
            if !normalized.is_empty() {
                state.blocks.push(ContentBlock {
                    kind: BlockKind::Paragraph,
                    text: normalized,
                    section_path: Vec::new(),
                    index: 0,
                });
            }
        }

        Ok(state.blocks)
    }
}

impl Default for StreamingExtractor {
    fn default() -> Self {
        StreamingExtractor::new()
    }
}

/// Convenience helper for in-memory HTML.
pub fn extract_blocks_streaming(html: &[u8]) -> Result<Vec<ContentBlock>, StreamError> {
    let mut extractor = StreamingExtractor::new();
    extractor.write(html)?;
    extractor.finish()
}

/// Errors surfaced while streaming HTML.
#[derive(Debug)]
pub enum StreamError {
    /// The HTML rewriter encountered malformed markup.
    Rewrite(lol_html::errors::RewritingError),
    /// Internal buffer still had outstanding references.
    CollectorInUse,
    /// Collector mutex was poisoned while draining results.
    CollectorPoisoned,
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rewrite(err) => write!(f, "html rewrite error: {err}"),
            Self::CollectorInUse => write!(f, "streaming collector still in use"),
            Self::CollectorPoisoned => write!(f, "streaming collector mutex poisoned"),
        }
    }
}

impl Error for StreamError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Rewrite(err) => Some(err),
            Self::CollectorInUse | Self::CollectorPoisoned => None,
        }
    }
}

const ROOT_MW_PARSER: u8 = 1;
const ROOT_MW_CONTENT: u8 = 2;
const ROOT_BODY_PARSER: u8 = 3;
const ROOT_BODY_CONTENT: u8 = 4;
const ROOT_BODY_PARSER_FALLBACK: u8 = 5;
const ROOT_BODY: u8 = 6;

struct StreamState {
    depth: usize,
    ignore_depth: usize,
    root_priority: u8,
    root_depth: Option<usize>,
    root_closed: bool,
    mw_content_depth: Option<usize>,
    body_content_depth: Option<usize>,
    body_depth: Option<usize>,
    blocks: Vec<ContentBlock>,
    stack: Vec<ActiveBlock>,
    fallback: String,
    headings: Vec<HeadingEntry>,
    next_index: usize,
}

impl StreamState {
    fn new() -> Self {
        Self {
            depth: 0,
            ignore_depth: 0,
            root_priority: u8::MAX,
            root_depth: None,
            root_closed: false,
            mw_content_depth: None,
            body_content_depth: None,
            body_depth: None,
            blocks: Vec::new(),
            stack: Vec::new(),
            fallback: String::new(),
            headings: Vec::new(),
            next_index: 0,
        }
    }

    fn in_root(&self) -> bool {
        match self.root_depth {
            Some(depth) => !self.root_closed && self.depth >= depth,
            None => false,
        }
    }

    fn reset_content(&mut self) {
        self.blocks.clear();
        self.stack.clear();
        self.fallback.clear();
        self.headings.clear();
        self.next_index = 0;
    }
}

struct ActiveBlock {
    kind: BlockKind,
    level: Option<u8>,
    text: String,
}

struct HeadingEntry {
    level: u8,
    text: String,
}

fn maybe_set_root(state: &mut StreamState, priority: u8, depth: usize) {
    if priority < state.root_priority {
        if state.root_priority != u8::MAX {
            state.reset_content();
        }
        state.root_closed = false;
        state.root_priority = priority;
        state.root_depth = Some(depth);
        debug_log!("root_depth set to {:?}", state.root_depth);
    }
}

fn should_ignore_tag(tag: &str) -> bool {
    if tag == "body" || tag == "html" {
        return false;
    }
    IGNORE_TAGS.contains(&tag)
}

fn should_ignore_class(class_name: &str) -> bool {
    if IGNORE_CLASSES.contains(&class_name) {
        return true;
    }
    IGNORE_CLASS_PREFIXES
        .iter()
        .any(|prefix| class_name.starts_with(prefix))
}

fn is_block_tag(tag: &str) -> bool {
    matches!(
        tag,
        "p" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "li" | "pre"
    )
}

fn finalize_block(state: &mut StreamState, block: ActiveBlock) -> Option<ContentBlock> {
    let decoded = decode_entities(&block.text);
    let normalized = match block.kind {
        BlockKind::Code => super::normalize_code(&decoded),
        _ => super::normalize_whitespace(&decoded),
    };
    if normalized.is_empty() {
        return None;
    }

    let section_path = if let Some(level) = block.level {
        update_heading_path(&mut state.headings, level, normalized.clone())
    } else {
        state
            .headings
            .iter()
            .map(|entry| entry.text.clone())
            .collect()
    };

    let block = ContentBlock {
        kind: block.kind,
        text: normalized,
        section_path,
        index: state.next_index,
    };
    state.next_index += 1;
    Some(block)
}

fn update_heading_path(headings: &mut Vec<HeadingEntry>, level: u8, text: String) -> Vec<String> {
    while headings.last().is_some_and(|entry| entry.level >= level) {
        headings.pop();
    }
    headings.push(HeadingEntry { level, text });
    headings.iter().map(|entry| entry.text.clone()).collect()
}

fn decode_entities(input: &str) -> String {
    decode_html_entities(input).into_owned()
}

struct NoopSink;

impl OutputSink for NoopSink {
    fn handle_chunk(&mut self, _chunk: &[u8]) {}
}

#[cfg(test)]
mod tests {
    use super::{extract_blocks_streaming, StreamingExtractor};
    use crate::{BlockKind, ContentBlock};

    #[test]
    fn extracts_streamed_blocks() {
        let html = r#"
        <div id="mw-content-text">
          <h2>Heading</h2>
          <p>Hello <sup>[1]</sup>world.</p>
          <pre>fn main() {
  println!("hi");
}</pre>
        </div>
        "#;

        let mut extractor = StreamingExtractor::new();
        for chunk in html.as_bytes().chunks(16) {
            extractor.write(chunk).unwrap();
        }
        let blocks = extractor.finish().unwrap();

        assert_eq!(
            blocks,
            vec![
                ContentBlock {
                    kind: BlockKind::Heading,
                    text: "Heading".to_string(),
                    section_path: vec!["Heading".to_string()],
                    index: 0,
                },
                ContentBlock {
                    kind: BlockKind::Paragraph,
                    text: "Hello world.".to_string(),
                    section_path: vec!["Heading".to_string()],
                    index: 1,
                },
                ContentBlock {
                    kind: BlockKind::Code,
                    text: "fn main() {\n  println!(\"hi\");\n}".to_string(),
                    section_path: vec!["Heading".to_string()],
                    index: 2,
                }
            ]
        );
    }

    #[test]
    fn falls_back_when_no_blocks_found() {
        let html = r#"<body><div>No blocks here.</div></body>"#;
        let blocks = extract_blocks_streaming(html.as_bytes()).unwrap();
        assert_eq!(
            blocks,
            vec![ContentBlock {
                kind: BlockKind::Paragraph,
                text: "No blocks here.".to_string(),
                section_path: Vec::new(),
                index: 0,
            }]
        );
    }
}
