use clap::Parser;
use fastcrawl::{runtime, Cli};
use std::sync::Arc;
use url::Url;

const SEEDS: &[&str] = &[
    // "https://en.wikipedia.org/wiki/Web_crawler",
    // "https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol",
    // "https://en.wikipedia.org/wiki/Capybara",
    // "https://en.wikipedia.org/wiki/Cat",
    "https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/1",
    "https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/2",
    "https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/3",
    "https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/4",
    "https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5",
];

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    runtime::run(cli, SEEDS, Arc::new(wiki_filter))
}

fn wiki_filter(url: &Url) -> bool {
    const ARTICLE_PREFIX: &str = "/wiki/";
    const BLOCKED_NAMESPACES: &[&str] = &[
        "Special:",
        "Help:",
        "Help_talk:",
        "Wikipedia:",
        "Talk:",
        "User:",
        "User_talk:",
        "File:",
        "Template:",
        "Portal:",
        "Portal_talk:",
        "Category:",
        "Book:",
        "Draft:",
        "MOS:",
    ];
    const BLOCKED_QUERY_SNIPPETS: &[&str] = &[
        "action=",
        "feed=",
        "printable=yes",
        "returnto=",
        "useskin=",
        "oldid=",
        "diff=",
        "curid=",
        "Special:",
    ];

    if url.scheme() != "https" || url.fragment().is_some() {
        return false;
    }

    let path = url.path();
    let Some(article) = path.strip_prefix(ARTICLE_PREFIX) else {
        return false;
    };

    if BLOCKED_NAMESPACES
        .iter()
        .any(|prefix| article.starts_with(prefix))
    {
        return false;
    }

    let blocked_query = url
        .query()
        .map(|query| {
            BLOCKED_QUERY_SNIPPETS
                .iter()
                .any(|blocked| query.contains(blocked))
        })
        .unwrap_or(false);

    !blocked_query
}
