//! HTML corpus normalization primitives for downstream LLM/RAG pipelines.

use crc32fast::Hasher as Crc32;
use reqwest::header::{HeaderMap, HeaderName, CONTENT_LANGUAGE, CONTENT_TYPE};
use scraper::{Html, Selector};
use serde::ser::{SerializeStruct, Serializer};
use serde::Serialize;
use std::borrow::Cow;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};
use url::Url;

/// Raw page bytes plus crawl metadata awaiting normalization.
#[derive(Debug, Clone)]
pub struct FetchedPage {
    /// Canonical URL of the fetched document.
    pub url: Url,
    /// Crawl depth when the page was scheduled.
    pub depth: u8,
    /// Timestamp when the fetch completed.
    pub fetched_at: SystemTime,
    /// HTTP response status code.
    pub status: u16,
    /// Response headers (already lowercased by Reqwest).
    pub headers: HeaderMap,
    /// Optional shard identifier when running in sharded mode.
    pub shard: Option<usize>,
    /// Raw response body bytes.
    pub body: Vec<u8>,
}

impl FetchedPage {
    /// Builds a new fetched page payload.
    pub fn new(
        url: Url,
        depth: u8,
        fetched_at: SystemTime,
        status: u16,
        headers: HeaderMap,
        body: Vec<u8>,
    ) -> Self {
        Self {
            url,
            depth,
            fetched_at,
            status,
            headers,
            shard: None,
            body,
        }
    }

    /// Annotates the page with its owning shard.
    pub fn with_shard(mut self, shard: usize) -> Self {
        self.shard = Some(shard);
        self
    }
}

/// Metadata captured during normalization.
#[derive(Debug, Clone)]
pub struct PageMetadata {
    /// Canonical URL.
    pub url: Url,
    /// Crawl depth.
    pub depth: u8,
    /// Optional shard affinity.
    pub shard: Option<usize>,
    /// Fetch completion timestamp.
    pub fetched_at: SystemTime,
    /// HTTP status code.
    pub status: u16,
    /// Content-Type header (if provided).
    pub content_type: Option<String>,
    /// Content-Language header (if provided).
    pub content_language: Option<String>,
    /// Body length in bytes.
    pub content_length: usize,
    /// CRC32 checksum of the raw body.
    pub checksum: u32,
    /// True when the body required lossy decoding.
    pub lossy_decoding: bool,
}

impl PageMetadata {
    fn from_page(page: &FetchedPage, lossy_decoding: bool) -> Self {
        let content_type = header_to_string(&page.headers, CONTENT_TYPE);
        let content_language = header_to_string(&page.headers, CONTENT_LANGUAGE);
        let content_length = page.body.len();

        let mut hasher = Crc32::new();
        hasher.update(&page.body);
        let checksum = hasher.finalize();

        Self {
            url: page.url.clone(),
            depth: page.depth,
            shard: page.shard,
            fetched_at: page.fetched_at,
            status: page.status,
            content_type,
            content_language,
            content_length,
            checksum,
            lossy_decoding,
        }
    }
}

impl Serialize for PageMetadata {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("PageMetadata", 10)?;
        state.serialize_field("url", &self.url)?;
        state.serialize_field("depth", &self.depth)?;
        state.serialize_field("shard", &self.shard)?;
        let fetched_ms = self
            .fetched_at
            .duration_since(UNIX_EPOCH)
            .map(|dur| dur.as_millis() as u64)
            .unwrap_or(0);
        state.serialize_field("fetched_at_epoch_ms", &fetched_ms)?;
        state.serialize_field("status", &self.status)?;
        state.serialize_field("content_type", &self.content_type)?;
        state.serialize_field("content_language", &self.content_language)?;
        state.serialize_field("content_length", &self.content_length)?;
        state.serialize_field("checksum", &self.checksum)?;
        state.serialize_field("lossy_decoding", &self.lossy_decoding)?;
        state.end()
    }
}

fn header_to_string(headers: &HeaderMap, name: HeaderName) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// Heading hierarchy captured while walking the document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SectionHeading {
    /// Heading depth (1-6).
    pub level: u8,
    /// Visible heading text.
    pub title: String,
}

/// Classification for extracted blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum BlockKind {
    /// Heading text at a given level.
    Heading {
        /// Heading depth (1-6).
        level: u8,
    },
    /// Standard paragraph.
    Paragraph,
    /// Bullet or numbered list item.
    ListItem,
    /// Inline or fenced preformatted snippet/code.
    Preformatted,
    /// Block quote.
    Quote,
}

/// Discrete chunk of cleaned text tagged with structural context.
#[derive(Debug, Clone, Serialize)]
pub struct TextBlock {
    /// Block classification.
    pub kind: BlockKind,
    /// Collapsed textual content.
    pub text: String,
    /// Character offset (in UTF-8 code units) within the normalized body.
    pub char_start: usize,
    /// Exclusive end offset within the normalized body.
    pub char_end: usize,
    /// Rough token estimate (word count heuristic).
    pub token_estimate: usize,
    /// Hierarchical headings leading to this block.
    pub section_path: Vec<SectionHeading>,
}

/// Chunk emitted for downstream embedding or indexing.
#[derive(Debug, Clone, Serialize)]
pub struct NormalizedChunk {
    /// Monotonic chunk identifier assigned during normalization.
    pub chunk_id: usize,
    /// Concatenated text for this chunk.
    pub text: String,
    /// Estimated token count.
    pub token_estimate: usize,
    /// Character start offset within the normalized body.
    pub char_start: usize,
    /// Character end offset within the normalized body.
    pub char_end: usize,
    /// Heading path representing the chunk's context.
    pub section_path: Vec<SectionHeading>,
}

/// Fully normalized representation of a fetched page.
#[derive(Debug, Clone, Serialize)]
pub struct NormalizedPage {
    /// Captured metadata.
    pub metadata: PageMetadata,
    /// Entire cleaned body text (blocks joined with double newlines).
    pub body_text: String,
    /// Intermediate structural blocks.
    pub blocks: Vec<TextBlock>,
    /// Final embedding-ready chunks.
    pub chunks: Vec<NormalizedChunk>,
}

/// Normalization tuning knobs.
#[derive(Debug, Clone, Copy)]
pub struct NormalizationConfig {
    /// Approximate tokens per chunk before flushing.
    pub chunk_target_tokens: usize,
    /// Desired tail-overlap between adjacent chunks (approximate tokens).
    pub chunk_overlap_tokens: usize,
    /// Cap on the number of recorded blocks to avoid runaway memory use.
    pub max_blocks: usize,
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            chunk_target_tokens: 256,
            chunk_overlap_tokens: 48,
            max_blocks: 8192,
        }
    }
}

/// Errors surfaced while normalizing a page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizationError {
    /// The response body was empty.
    EmptyBody,
}

impl fmt::Display for NormalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBody => write!(f, "no body bytes available for normalization"),
        }
    }
}

impl std::error::Error for NormalizationError {}

/// Stateless HTML normalization service.
#[derive(Clone)]
pub struct Normalizer {
    config: NormalizationConfig,
    selectors: RootSelectors,
}

impl Normalizer {
    /// Builds a new normalizer instance.
    pub fn new(config: NormalizationConfig) -> Self {
        Self {
            config,
            selectors: RootSelectors::new(),
        }
    }

    /// Returns the underlying config reference.
    pub fn config(&self) -> &NormalizationConfig {
        &self.config
    }

    /// Normalizes a fetched page into cleaned text blocks and chunks.
    pub fn normalize(&self, page: &FetchedPage) -> Result<NormalizedPage, NormalizationError> {
        if page.body.is_empty() {
            return Err(NormalizationError::EmptyBody);
        }

        let (decoded, lossy) = decode_body(&page.body);
        let document = Html::parse_document(&decoded);
        let root = self.selectors.pick_root(&document);

        let mut collector = BlockCollector::new(&self.config);
        collector.walk(root);
        let (body_text, blocks) = collector.finish();

        let chunks = chunk_blocks(&blocks, &self.config);
        let metadata = PageMetadata::from_page(page, lossy);

        Ok(NormalizedPage {
            metadata,
            body_text,
            blocks,
            chunks,
        })
    }
}

#[derive(Clone)]
struct RootSelectors {
    article: Selector,
    main: Selector,
    body: Selector,
}

impl RootSelectors {
    fn new() -> Self {
        Self {
            article: Selector::parse("article").expect("article selector"),
            main: Selector::parse("main").expect("main selector"),
            body: Selector::parse("body").expect("body selector"),
        }
    }

    fn pick_root<'a>(&self, document: &'a Html) -> scraper::ElementRef<'a> {
        document
            .select(&self.article)
            .next()
            .or_else(|| document.select(&self.main).next())
            .or_else(|| document.select(&self.body).next())
            .unwrap_or_else(|| document.root_element())
    }
}

fn decode_body(bytes: &[u8]) -> (Cow<'_, str>, bool) {
    match std::str::from_utf8(bytes) {
        Ok(text) => (Cow::Borrowed(text), false),
        Err(_) => (
            Cow::Owned(String::from_utf8_lossy(bytes).into_owned()),
            true,
        ),
    }
}

struct BlockCollector<'cfg> {
    config: &'cfg NormalizationConfig,
    body_text: String,
    blocks: Vec<TextBlock>,
    current_path: Vec<SectionHeading>,
    block_limit_hit: bool,
}

impl<'cfg> BlockCollector<'cfg> {
    fn new(config: &'cfg NormalizationConfig) -> Self {
        Self {
            config,
            body_text: String::new(),
            blocks: Vec::new(),
            current_path: Vec::new(),
            block_limit_hit: false,
        }
    }

    fn walk(&mut self, root: scraper::ElementRef<'_>) {
        if self.block_limit_hit {
            return;
        }
        for element in root.descendent_elements() {
            if self.block_limit_hit {
                break;
            }
            self.maybe_record(element);
        }
    }

    fn maybe_record(&mut self, element: scraper::ElementRef<'_>) {
        let tag = element.value().name();
        if matches!(
            tag,
            "script" | "style" | "template" | "noscript" | "svg" | "nav"
        ) {
            return;
        }

        let kind = match tag {
            "h1" => Some(BlockKind::Heading { level: 1 }),
            "h2" => Some(BlockKind::Heading { level: 2 }),
            "h3" => Some(BlockKind::Heading { level: 3 }),
            "h4" => Some(BlockKind::Heading { level: 4 }),
            "h5" => Some(BlockKind::Heading { level: 5 }),
            "h6" => Some(BlockKind::Heading { level: 6 }),
            "p" => Some(BlockKind::Paragraph),
            "li" => Some(BlockKind::ListItem),
            "blockquote" => Some(BlockKind::Quote),
            "pre" | "code" => Some(BlockKind::Preformatted),
            _ => None,
        };

        let Some(kind) = kind else {
            return;
        };

        let preserve_newlines = matches!(kind, BlockKind::Preformatted);
        let text = extract_text(&element, preserve_newlines);
        if text.is_empty() {
            return;
        }

        if let BlockKind::Heading { level } = kind {
            self.update_heading_path(level, &text);
            self.push_block(BlockKind::Heading { level }, text);
        } else {
            self.push_block(kind, text);
        }

        if self.blocks.len() >= self.config.max_blocks {
            self.block_limit_hit = true;
        }
    }

    fn update_heading_path(&mut self, level: u8, text: &str) {
        while let Some(last) = self.current_path.last() {
            if last.level >= level {
                self.current_path.pop();
            } else {
                break;
            }
        }
        self.current_path.push(SectionHeading {
            level,
            title: text.to_string(),
        });
    }

    fn push_block(&mut self, kind: BlockKind, text: String) {
        if !self.body_text.is_empty() {
            self.body_text.push_str("\n\n");
        }
        let start = self.body_text.len();
        self.body_text.push_str(&text);
        let end = self.body_text.len();
        let tokens = estimate_tokens(&text);
        self.blocks.push(TextBlock {
            kind,
            text,
            char_start: start,
            char_end: end,
            token_estimate: tokens,
            section_path: self.current_path.clone(),
        });
    }

    fn finish(self) -> (String, Vec<TextBlock>) {
        (self.body_text, self.blocks)
    }
}

fn extract_text(element: &scraper::ElementRef<'_>, preserve_newlines: bool) -> String {
    let mut raw = String::new();
    for piece in element.text() {
        raw.push_str(piece);
    }
    if preserve_newlines {
        collapse_newlines(&raw)
    } else {
        collapse_whitespace(&raw)
    }
}

fn collapse_whitespace(input: &str) -> String {
    let mut buf = String::with_capacity(input.len());
    let mut last_space = false;
    for ch in input.chars() {
        if ch.is_whitespace() {
            if !last_space && !buf.is_empty() {
                buf.push(' ');
            }
            last_space = true;
        } else {
            buf.push(ch);
            last_space = false;
        }
    }
    buf.trim().to_string()
}

fn collapse_newlines(input: &str) -> String {
    let mut lines = Vec::new();
    for line in input.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        lines.push(trimmed.to_string());
    }
    lines.join("\n")
}

fn estimate_tokens(text: &str) -> usize {
    let tokens = text.split_whitespace().count();
    tokens.max(1)
}

fn chunk_blocks(blocks: &[TextBlock], config: &NormalizationConfig) -> Vec<NormalizedChunk> {
    if blocks.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut buffer: Vec<usize> = Vec::new();
    let mut token_total = 0usize;
    let target = config.chunk_target_tokens.max(1);
    let overlap = config.chunk_overlap_tokens.min(target.saturating_sub(1));

    for (idx, block) in blocks.iter().enumerate() {
        if block.text.is_empty() {
            continue;
        }
        buffer.push(idx);
        token_total += block.token_estimate.max(1);

        if token_total >= target {
            flush_chunk(&mut chunks, &buffer, blocks);
            if overlap == 0 {
                buffer.clear();
                token_total = 0;
            } else {
                buffer = retain_overlap(&buffer, blocks, overlap);
                token_total = buffer
                    .iter()
                    .map(|&idx| blocks[idx].token_estimate.max(1))
                    .sum();
            }
        }
    }

    if !buffer.is_empty() {
        flush_chunk(&mut chunks, &buffer, blocks);
    }

    chunks
}

fn flush_chunk(chunks: &mut Vec<NormalizedChunk>, buffer: &[usize], blocks: &[TextBlock]) {
    if buffer.is_empty() {
        return;
    }

    let mut text = String::new();
    let mut token_estimate = 0usize;
    for (i, &block_idx) in buffer.iter().enumerate() {
        if i > 0 {
            text.push_str("\n\n");
        }
        text.push_str(&blocks[block_idx].text);
        token_estimate += blocks[block_idx].token_estimate;
    }

    let first = buffer.first().copied().unwrap();
    let last = buffer.last().copied().unwrap();
    let section_path = blocks[last].section_path.clone();

    chunks.push(NormalizedChunk {
        chunk_id: chunks.len(),
        text,
        token_estimate,
        char_start: blocks[first].char_start,
        char_end: blocks[last].char_end,
        section_path,
    });
}

fn retain_overlap(buffer: &[usize], blocks: &[TextBlock], overlap: usize) -> Vec<usize> {
    if overlap == 0 || buffer.is_empty() {
        return Vec::new();
    }
    let mut retained = Vec::new();
    let mut tokens = 0usize;
    for &idx in buffer.iter().rev() {
        retained.push(idx);
        tokens += blocks[idx].token_estimate;
        if tokens >= overlap {
            break;
        }
    }
    retained.reverse();
    retained
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;

    #[test]
    fn normalizes_basic_article() {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/html"));
        let body = br#"
            <html>
              <body>
                <article>
                  <h1>Example</h1>
                  <p>First paragraph with <b>bold</b> text.</p>
                  <h2>Details</h2>
                  <p>More info.</p>
                </article>
              </body>
            </html>
        "#
        .to_vec();

        let page = FetchedPage::new(
            Url::parse("https://example.com").unwrap(),
            1,
            SystemTime::now(),
            200,
            headers,
            body,
        );

        let normalizer = Normalizer::new(NormalizationConfig::default());
        let page = normalizer.normalize(&page).expect("normalize");

        assert_eq!(page.blocks.len(), 4);
        assert_eq!(page.chunks.len(), 1);
        assert!(page.body_text.contains("First paragraph"));
        assert_eq!(page.chunks[0].section_path.last().unwrap().title, "Details");
    }

    #[test]
    fn respects_chunk_target() {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/html"));
        let mut body = String::from("<article><h1>Title</h1>");
        for _ in 0..20 {
            body.push_str("<p>Lorem ipsum dolor sit amet.</p>");
        }
        body.push_str("</article>");

        let page = FetchedPage::new(
            Url::parse("https://example.com/long").unwrap(),
            1,
            SystemTime::now(),
            200,
            headers,
            body.into_bytes(),
        );

        let mut config = NormalizationConfig::default();
        config.chunk_target_tokens = 30;
        config.chunk_overlap_tokens = 10;
        let normalizer = Normalizer::new(config);
        let normalized = normalizer.normalize(&page).expect("normalize");

        assert!(normalized.chunks.len() > 1);
        assert!(normalized.chunks.windows(2).all(|w| {
            let a = &w[0];
            let b = &w[1];
            a.char_end > b.char_start || a.char_end == b.char_start
        }));
    }
}
