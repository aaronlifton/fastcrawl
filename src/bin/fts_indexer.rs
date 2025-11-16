use anyhow::{Context, Result};
use clap::Parser;
use fastcrawl::TableName;
use std::io::{self, Write};
use tokio_postgres::{Client, NoTls};

#[derive(Parser, Debug)]
#[command(
    name = "fastcrawl-fts-indexer",
    about = "Ensure text_tsv columns and GIN indexes exist for lexical search"
)]
struct FtsCli {
    /// Postgres connection string (postgres://...)
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,

    /// Target schema for the vector table
    #[arg(long, env = "FASTCRAWL_PGVECTOR_SCHEMA", default_value = "public")]
    schema: String,

    /// Target table name inside the schema
    #[arg(long, env = "FASTCRAWL_PGVECTOR_TABLE", default_value = "chunks")]
    table: String,

    /// Full-text language passed to to_tsvector
    #[arg(long, env = "FASTCRAWL_FTS_LANGUAGE", default_value = "english")]
    language: String,

    /// Analyze the table after ensuring indexes
    #[arg(long, default_value_t = true)]
    analyze: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = FtsCli::parse();
    let table = TableName::new(cli.schema, cli.table)?;
    let language = cli.language.trim();
    anyhow::ensure!(!language.is_empty(), "language must not be empty");

    let (client, connection) = tokio_postgres::connect(&cli.database_url, NoTls)
        .await
        .with_context(|| format!("failed to connect to Postgres at {}", cli.database_url))?;
    tokio::spawn(async move {
        if let Err(err) = connection.await {
            eprintln!("postgres connection error: {err}");
        }
    });
    let mut client = client;

    let total_steps = 2 + usize::from(cli.analyze);
    let mut completed = 0usize;

    render_progress(completed, total_steps, "ensuring text_tsv column");
    ensure_fts_column(&mut client, &table, language).await?;
    completed += 1;
    render_progress(completed, total_steps, "creating GIN index");
    ensure_fts_index(&mut client, &table).await?;
    completed += 1;
    if cli.analyze {
        render_progress(completed, total_steps, "running ANALYZE");
        analyze_table(&mut client, &table).await?;
        completed += 1;
    }
    render_progress(completed, total_steps, "complete");
    if total_steps > 0 {
        println!();
    }
    println!(
        "Ensured text_tsv column and GIN index exist on {}.",
        table.qualified()
    );
    Ok(())
}

async fn ensure_fts_column(client: &mut Client, table: &TableName, language: &str) -> Result<()> {
    let literal = escape_literal(language);
    let alter = format!(
        "ALTER TABLE {} ADD COLUMN IF NOT EXISTS text_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('{}', text)) STORED",
        table.qualified(),
        literal
    );
    client
        .execute(&alter, &[])
        .await
        .context("failed to ensure text_tsv column")?;
    Ok(())
}

async fn ensure_fts_index(client: &mut Client, table: &TableName) -> Result<()> {
    let index_name = table.fts_index_name();
    let sql = format!(
        "CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (text_tsv)",
        index_name,
        table.qualified()
    );
    client
        .execute(&sql, &[])
        .await
        .context("failed to ensure text_tsv GIN index")?;
    Ok(())
}

async fn analyze_table(client: &mut Client, table: &TableName) -> Result<()> {
    let sql = format!("ANALYZE {}", table.qualified());
    client
        .execute(&sql, &[])
        .await
        .context("failed to analyze table")?;
    Ok(())
}

fn escape_literal(input: &str) -> String {
    input.replace('\'', "''")
}

fn render_progress(done: usize, total: usize, label: &str) {
    if total == 0 {
        return;
    }
    let pct = (done as f64 / total as f64) * 100.0;
    print!("\r[{}/{}] {:.1}% {}", done, total, pct, label);
    let _ = io::stdout().flush();
}
