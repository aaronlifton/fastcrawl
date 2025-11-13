//! Streaming HTML helpers built on `lol_html`.

use futures_util::StreamExt;
use lol_html::{element, HtmlRewriter, OutputSink, Settings};
use reqwest::Response;
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};

/// Result of streaming link extraction.
pub struct LinkHarvest<T> {
    /// Accepted links transformed by the caller.
    pub links: Vec<T>,
    /// Optional raw body bytes captured during streaming.
    pub body: Option<Vec<u8>>,
}

/// Streams anchor tags from an HTTP response, transforming matching `href` values with `transform`.
///
/// The `transform` closure runs for every `href`; returning `Some(T)` keeps the value, `None` skips
/// it. Only accepted entries count against `limit`. When `capture_body` is true the full response
/// body is buffered and returned alongside the discovered links.
pub async fn stream_links<T, F>(
    response: Response,
    limit: usize,
    capture_body: bool,
    transform: F,
) -> Result<LinkHarvest<T>, HtmlStreamError>
where
    T: Send + 'static,
    F: Fn(&str) -> Option<T> + Send + Sync + 'static,
{
    if limit == 0 && !capture_body {
        return Ok(LinkHarvest {
            links: Vec::new(),
            body: None,
        });
    }

    let values: Arc<Mutex<Vec<T>>> = Arc::new(Mutex::new(Vec::new()));
    let values_handle = Arc::clone(&values);
    let transform = Arc::new(transform);
    let transform_handle = Arc::clone(&transform);

    let handler = element!("a[href]", move |el| {
        let mut entries = values_handle
            .lock()
            .unwrap_or_else(|_| panic!("link collector mutex poisoned"));
        if entries.len() >= limit {
            return Ok(());
        }

        if let Some(href) = el.get_attribute("href") {
            if let Some(mapped) = transform_handle(&href) {
                entries.push(mapped);
            }
        }
        Ok(())
    });

    let mut rewriter = HtmlRewriter::new(
        Settings {
            element_content_handlers: vec![handler],
            ..Settings::default()
        },
        NoopSink,
    );

    let mut stream = response.bytes_stream();
    let mut body_buf = capture_body.then(Vec::new);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(HtmlStreamError::Http)?;
        if let Some(buf) = body_buf.as_mut() {
            buf.extend_from_slice(&chunk);
        }
        rewriter.write(&chunk).map_err(HtmlStreamError::Rewrite)?;
    }
    rewriter.end().map_err(HtmlStreamError::Rewrite)?;

    drop(transform);

    let collected = Arc::try_unwrap(values)
        .map_err(|_| HtmlStreamError::CollectorInUse)?
        .into_inner()
        .map_err(|_| HtmlStreamError::CollectorPoisoned)?;

    Ok(LinkHarvest {
        links: collected,
        body: body_buf,
    })
}

/// Errors surfaced while streaming HTML.
#[derive(Debug)]
pub enum HtmlStreamError {
    /// Reading the response stream failed.
    Http(reqwest::Error),
    /// The HTML rewriter encountered malformed markup.
    Rewrite(lol_html::errors::RewritingError),
    /// Internal buffer still had outstanding references.
    CollectorInUse,
    /// Collector mutex was poisoned while draining results.
    CollectorPoisoned,
}

impl fmt::Display for HtmlStreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(err) => write!(f, "http stream error: {err}"),
            Self::Rewrite(err) => write!(f, "html rewrite error: {err}"),
            Self::CollectorInUse => write!(f, "link collector still in use"),
            Self::CollectorPoisoned => write!(f, "link collector mutex poisoned"),
        }
    }
}

impl Error for HtmlStreamError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Http(err) => Some(err),
            Self::Rewrite(err) => Some(err),
            Self::CollectorInUse | Self::CollectorPoisoned => None,
        }
    }
}

struct NoopSink;

impl OutputSink for NoopSink {
    fn handle_chunk(&mut self, _chunk: &[u8]) {}
}
