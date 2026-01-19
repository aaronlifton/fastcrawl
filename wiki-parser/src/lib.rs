//! Extract readable text from Wikipedia HTML pages.
//!
//! This crate focuses on pulling main article text while skipping common
//! non-content elements like navigation boxes, infoboxes, and references.

use ego_tree::NodeRef;
use scraper::{node::Node, ElementRef, Html, Selector};
use serde::Serialize;

mod stream;

// This enables or disables the eprintln! macro, effectively removing it from the code when the
// "debug" feature is disabled.
#[cfg(feature = "debug_logs")]
#[macro_export]
macro_rules! debug_log {
        ($($arg:tt)*) => {
            eprintln!($($arg)*);
        };
    }
#[cfg(not(feature = "debug_logs"))]
#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {};
}

const IGNORE_TAGS: &[&str] = &[
    "script", "style", "table", "sup", "noscript", "figure", "header", "footer", "nav", "aside",
];

const IGNORE_CLASSES: &[&str] = &[
    "ambox",
    "authority-control",
    "autocollapse",
    "catlinks",
    "cmbox",
    "collapsible-list",
    "dablink",
    "fmbox",
    "hatnote",
    "imbox",
    "infobox",
    "infobox-above",
    "infobox-caption",
    "infobox-data",
    "infobox-full-data",
    "infobox-header",
    "infobox-image",
    "infobox-label",
    "infobox-signature",
    "infobox-subheader",
    "infobox-title",
    "mbox-image",
    "mbox-small",
    "mbox-text",
    "metadata",
    "messagebox",
    "multiple-issues",
    "mw-collapsible",
    "mw-collapsible-content",
    "mw-editsection",
    "mw-editsection-bracket",
    "mw-footer",
    "mw-footer-container",
    "mw-hidden-catlinks",
    "mw-normal-catlinks",
    "mw-portlet-sticky-header-toc",
    "mw-references-columns",
    "mw-references-wrap",
    "navbox",
    "navbox-abovebelow",
    "navbox-columns-table",
    "navbox-even",
    "navbox-group",
    "navbox-image",
    "navbox-inner",
    "navbox-list",
    "navbox-list-with-group",
    "navbox-odd",
    "navbox-styles",
    "navbox-subgroup",
    "navbox-title",
    "nowraplinks",
    "ombox",
    "pie-thumb",
    "printfooter",
    "refbegin",
    "reference",
    "reference-text",
    "reflist",
    "references",
    "references-column-width",
    "shortdescription",
    "sidebar",
    "sidebar-content",
    "sidebar-navbar",
    "sidebar-title",
    "thumb",
    "thumbcaption",
    "thumbimage",
    "thumbinner",
    "toc",
    "toccolours",
    "vector-page-titlebar-toc",
    "vector-sticky-header-toc",
    "vertical-navbox",
];

const IGNORE_CLASS_PREFIXES: &[&str] = &["vector-toc", "toclimit-"];

/// A structured block of extracted content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockKind {
    Paragraph,
    Heading,
    ListItem,
    Code,
}

/// A structured block of extracted content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub kind: BlockKind,
    pub text: String,
    pub section_path: Vec<String>,
    pub index: usize,
}

pub use stream::{extract_blocks_streaming, StreamError, StreamingExtractor};

/// Extracts readable text blocks from a Wikipedia HTML document.
///
/// # Example
///
/// ```
/// use wiki_parser::extract_blocks;
///
/// let html = r#"<div id=\"mw-content-text\"><p>Hello <sup>[1]</sup>world.</p></div>"#;
/// let blocks = extract_blocks(html);
/// assert_eq!(blocks.len(), 1);
/// assert_eq!(blocks[0].text, "Hello world.");
/// ```
pub fn extract_blocks(html: &str) -> Vec<ContentBlock> {
    let document = Html::parse_document(html);
    let root = select_content_root(&document).unwrap_or_else(|| document.root_element());

    let mut blocks = Vec::new();
    let mut headings: Vec<HeadingEntry> = Vec::new();
    let mut next_index = 0usize;
    let selector = Selector::parse("p, h1, h2, h3, h4, h5, h6, li, pre")
        .expect("valid selector for content nodes");
    for node in root.select(&selector) {
        if has_ignored_ancestor(&node) {
            continue;
        }
        let mut buf = String::new();
        collect_text(&node, &mut buf);
        let kind = classify_block(node.value().name());
        let level = heading_level(node.value().name());
        let normalized = match kind {
            BlockKind::Code => normalize_code(&buf),
            _ => normalize_whitespace(&buf),
        };
        if !normalized.is_empty() {
            let section_path = if let Some(level) = level {
                update_heading_path(&mut headings, level, normalized.clone())
            } else {
                headings.iter().map(|entry| entry.text.clone()).collect()
            };
            blocks.push(ContentBlock {
                kind,
                text: normalized,
                section_path,
                index: next_index,
            });
            next_index += 1;
        }
    }

    if blocks.is_empty() {
        let mut buf = String::new();
        collect_text(&root, &mut buf);
        let normalized = normalize_whitespace(&buf);
        if !normalized.is_empty() {
            blocks.push(ContentBlock {
                kind: BlockKind::Paragraph,
                text: normalized,
                section_path: Vec::new(),
                index: 0,
            });
        }
    }

    blocks
}

/// Extracts readable text from a Wikipedia HTML document.
///
/// # Example
///
/// ```
/// use wiki_parser::extract_text;
///
/// let html = r#"<div id=\"mw-content-text\"><p>Hello <sup>[1]</sup>world.</p></div>"#;
/// assert_eq!(extract_text(html), "Hello world.");
/// ```
pub fn extract_text(html: &str) -> String {
    let blocks = extract_blocks(html);
    blocks
        .into_iter()
        .map(|block| block.text)
        .collect::<Vec<_>>()
        .join("\n")
}

fn select_content_root(document: &Html) -> Option<ElementRef<'_>> {
    let selectors = [
        "#mw-content-text .mw-parser-output",
        "#mw-content-text",
        "#bodyContent .mw-parser-output",
        "#bodyContent",
        "body .mw-parser-output",
        "body",
    ];
    for selector in selectors {
        let parsed = Selector::parse(selector).expect("valid selector");
        if let Some(node) = document.select(&parsed).next() {
            return Some(node);
        }
    }
    None
}

fn collect_text(node: &NodeRef<'_, Node>, out: &mut String) {
    match node.value() {
        Node::Text(text) => {
            out.push_str(text);
        }
        Node::Element(element) => {
            if should_ignore_element(element) {
                return;
            }
            for child in node.children() {
                collect_text(&child, out);
            }
        }
        _ => {
            for child in node.children() {
                collect_text(&child, out);
            }
        }
    }
}

fn should_ignore_element(element: &scraper::node::Element) -> bool {
    let tag_name = element.name();
    if tag_name == "body" || tag_name == "html" {
        return false;
    }
    if IGNORE_TAGS.contains(&tag_name) {
        return true;
    }
    for class_name in element.classes() {
        if IGNORE_CLASSES.contains(&class_name)
            || IGNORE_CLASS_PREFIXES
                .iter()
                .any(|prefix| class_name.starts_with(prefix))
        {
            return true;
        }
    }
    false
}

fn has_ignored_ancestor(node: &ElementRef<'_>) -> bool {
    if should_ignore_element(node.value()) {
        return true;
    }
    for ancestor in node.ancestors() {
        if let Some(element) = ElementRef::wrap(ancestor) {
            if should_ignore_element(element.value()) {
                return true;
            }
        }
    }
    false
}

fn normalize_whitespace(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut last_was_space = false;
    for ch in input.chars() {
        if ch.is_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    out.trim().to_string()
}

fn normalize_code(input: &str) -> String {
    input.trim().to_string()
}

fn classify_block(tag: &str) -> BlockKind {
    match tag {
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => BlockKind::Heading,
        "li" => BlockKind::ListItem,
        "pre" => BlockKind::Code,
        _ => BlockKind::Paragraph,
    }
}

fn heading_level(tag: &str) -> Option<u8> {
    match tag {
        "h1" => Some(1),
        "h2" => Some(2),
        "h3" => Some(3),
        "h4" => Some(4),
        "h5" => Some(5),
        "h6" => Some(6),
        _ => None,
    }
}

fn update_heading_path(headings: &mut Vec<HeadingEntry>, level: u8, text: String) -> Vec<String> {
    while headings.last().is_some_and(|entry| entry.level >= level) {
        headings.pop();
    }
    headings.push(HeadingEntry { level, text });
    headings.iter().map(|entry| entry.text.clone()).collect()
}

struct HeadingEntry {
    level: u8,
    text: String,
}

#[cfg(test)]
mod tests {
    use super::{extract_blocks, extract_text, BlockKind, ContentBlock};
    use pretty_assertions::assert_eq;

    #[test]
    fn extracts_paragraph_text() {
        let html = r#"
        <html>
          <body>
            <div id="mw-content-text">
              <p>Hello <sup>[1]</sup>world.</p>
              <p>Second paragraph.</p>
            </div>
          </body>
        </html>
        "#;

        assert_eq!(extract_text(html), "Hello world.\nSecond paragraph.");
    }

    #[test]
    fn extracts_blocks_with_types() {
        let html = r#"
        <div id="mw-content-text">
          <h2>Heading</h2>
          <p>Paragraph text.</p>
          <pre>fn main() {
  println!("hi");
}</pre>
          <ul><li>Item one</li></ul>
        </div>
        "#;

        let blocks = extract_blocks(html);
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
                    text: "Paragraph text.".to_string(),
                    section_path: vec!["Heading".to_string()],
                    index: 1,
                },
                ContentBlock {
                    kind: BlockKind::Code,
                    text: "fn main() {\n  println!(\"hi\");\n}".to_string(),
                    section_path: vec!["Heading".to_string()],
                    index: 2,
                },
                ContentBlock {
                    kind: BlockKind::ListItem,
                    text: "Item one".to_string(),
                    section_path: vec!["Heading".to_string()],
                    index: 3,
                }
            ]
        );
    }

    #[test]
    fn skips_infobox_and_navbox() {
        let html = r#"
        <div id="mw-content-text">
          <table class="infobox"><tr><td>Ignore</td></tr></table>
          <p>Keep this.</p>
          <div class="navbox">Also ignore</div>
          <ul><li>Item one</li><li>Item two</li></ul>
        </div>
        "#;

        assert_eq!(extract_text(html), "Keep this.\nItem one\nItem two");
    }

    #[test]
    fn falls_back_to_body_when_missing_content_div() {
        let html = r#"<body><p>Body text</p></body>"#;
        assert_eq!(extract_text(html), "Body text");
    }
}
