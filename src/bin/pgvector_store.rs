use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use fastcrawl::EmbeddedChunkRecord;
use pgvector::Vector;
use tokio_postgres::types::Json;
use tokio_postgres::{Client, NoTls};

#[derive(Parser, Debug)]
#[command(
    name = "fastcrawl-pgvector",
    about = "Load embedded chunks into a pgvector-backed Postgres table"
)]
struct PgVectorCli {
    /// Path to the JSONL file produced by fastcrawl-embedder
    #[arg(
        long,
        env = "FASTCRAWL_EMBED_OUTPUT",
        default_value = "embeddings.jsonl"
    )]
    input: PathBuf,

    /// Postgres connection string (postgres://...)
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,

    /// Target schema for the vector table
    #[arg(long, env = "FASTCRAWL_PGVECTOR_SCHEMA", default_value = "public")]
    schema: String,

    /// Target table name inside the schema
    #[arg(long, env = "FASTCRAWL_PGVECTOR_TABLE", default_value = "chunks")]
    table: String,

    /// Number of rows buffered per INSERT transaction
    #[arg(long, env = "FASTCRAWL_PGVECTOR_BATCH", default_value_t = 128)]
    batch_size: usize,

    /// Create the vector extension/table automatically if missing
    #[arg(long, env = "FASTCRAWL_PGVECTOR_PREPARE", default_value_t = true)]
    prepare_table: bool,

    /// Upsert rows when (url, chunk_id) already exists
    #[arg(long, env = "FASTCRAWL_PGVECTOR_UPSERT", default_value_t = true)]
    upsert: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = PgVectorCli::parse();
    let batch_size = cli.batch_size.max(1);
    let file = File::open(&cli.input)
        .with_context(|| format!("failed to open embedding input {:?}", cli.input))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines().enumerate();

    let (client, connection) = tokio_postgres::connect(&cli.database_url, NoTls)
        .await
        .with_context(|| format!("failed to connect to Postgres at {}", cli.database_url))?;
    tokio::spawn(async move {
        if let Err(err) = connection.await {
            eprintln!("postgres connection error: {err}");
        }
    });
    let mut client = client;

    let mut batch = Vec::with_capacity(batch_size);
    let Some(first_record) = next_record(&mut lines)? else {
        println!("No embeddings to insert; nothing to do.");
        return Ok(());
    };
    anyhow::ensure!(
        !first_record.embedding.is_empty(),
        "first embedding record missing vector values"
    );

    let table = TableName::new(cli.schema, cli.table)?;
    let dims = first_record.embedding.len();
    if cli.prepare_table {
        ensure_vector_extension(&mut client).await?;
        ensure_table(&mut client, &table, dims).await?;
    }

    batch.push(first_record);
    let mut total_inserted = 0usize;
    while let Some(record) = next_record(&mut lines)? {
        batch.push(record);
        if batch.len() >= batch_size {
            insert_batch(&mut client, &table, &batch, cli.upsert).await?;
            total_inserted += batch.len();
            batch.clear();
        }
    }

    if !batch.is_empty() {
        insert_batch(&mut client, &table, &batch, cli.upsert).await?;
        total_inserted += batch.len();
    }

    println!(
        "Successfully inserted {} record{} into {}.",
        total_inserted,
        if total_inserted == 1 { "" } else { "s" },
        table.qualified()
    );

    Ok(())
}

async fn ensure_vector_extension(client: &mut Client) -> Result<()> {
    client
        .execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
        .await
        .context("failed to ensure pgvector extension")?;
    Ok(())
}

async fn ensure_table(client: &mut Client, table: &TableName, dims: usize) -> Result<()> {
    anyhow::ensure!(dims > 0, "embedding dimension must be positive");
    let ddl = format!(
        "CREATE TABLE IF NOT EXISTS {} (
            url TEXT NOT NULL,
            chunk_id BIGINT NOT NULL,
            text TEXT NOT NULL,
            section_path JSONB NOT NULL,
            token_estimate BIGINT NOT NULL,
            embedding VECTOR({dims}) NOT NULL,
            checksum BIGINT NOT NULL,
            last_seen_epoch_ms BIGINT NOT NULL,
            PRIMARY KEY (url, chunk_id)
        )",
        table.qualified()
    );
    client
        .execute(&ddl, &[])
        .await
        .context("failed to create pgvector table")?;
    Ok(())
}

async fn insert_batch(
    client: &mut Client,
    table: &TableName,
    records: &[EmbeddedChunkRecord],
    upsert: bool,
) -> Result<()> {
    if records.is_empty() {
        return Ok(());
    }

    let sql = insert_sql(table, upsert);
    let transaction = client.transaction().await?;
    let statement = transaction.prepare(&sql).await?;
    for record in records {
        let vector = Vector::from(record.embedding.clone());
        let section_path = Json(record.section_path.clone());
        let chunk_id = as_i64(record.chunk_id, "chunk_id")?;
        let token_estimate = as_i64(record.token_estimate, "token_estimate")?;
        let checksum = as_i64(record.checksum, "checksum")?;
        let last_seen = as_i64(record.last_seen_epoch_ms, "last_seen_epoch_ms")?;
        transaction
            .execute(
                &statement,
                &[
                    &record.url,
                    &chunk_id,
                    &record.text,
                    &section_path,
                    &token_estimate,
                    &vector,
                    &checksum,
                    &last_seen,
                ],
            )
            .await
            .with_context(|| {
                format!(
                    "failed to insert chunk {} from {}",
                    record.chunk_id, record.url
                )
            })?;
    }
    transaction.commit().await?;
    Ok(())
}

fn insert_sql(table: &TableName, upsert: bool) -> String {
    let qualified = table.qualified();
    let mut sql = format!(
        "INSERT INTO {} \
            (url, chunk_id, text, section_path, token_estimate, embedding, checksum, last_seen_epoch_ms) \
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        qualified
    );
    if upsert {
        sql.push_str(
            " ON CONFLICT (url, chunk_id) DO UPDATE SET \
                text = EXCLUDED.text, \
                section_path = EXCLUDED.section_path, \
                token_estimate = EXCLUDED.token_estimate, \
                embedding = EXCLUDED.embedding, \
                checksum = EXCLUDED.checksum, \
                last_seen_epoch_ms = EXCLUDED.last_seen_epoch_ms",
        );
    }
    sql
}

fn next_record<I>(lines: &mut I) -> Result<Option<EmbeddedChunkRecord>>
where
    I: Iterator<Item = (usize, std::io::Result<String>)>,
{
    while let Some((line_no, line)) = lines.next() {
        let line = line.with_context(|| format!("failed to read line {}", line_no + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: EmbeddedChunkRecord = serde_json::from_str(&line)
            .with_context(|| format!("invalid embedding record at line {}", line_no + 1))?;
        return Ok(Some(record));
    }
    Ok(None)
}

fn as_i64<T>(value: T, field: &str) -> Result<i64>
where
    i64: TryFrom<T>,
    T: Copy + std::fmt::Display,
{
    i64::try_from(value).map_err(|_| anyhow!("{} value {} exceeds i64 range", field, value))
}

struct TableName {
    schema: String,
    table: String,
}

impl TableName {
    fn new(schema: String, table: String) -> Result<Self> {
        anyhow::ensure!(!schema.trim().is_empty(), "schema name is required");
        anyhow::ensure!(!table.trim().is_empty(), "table name is required");
        Ok(Self { schema, table })
    }

    fn qualified(&self) -> String {
        format!("{}.{}", quote_ident(&self.schema), quote_ident(&self.table))
    }
}

fn quote_ident(input: &str) -> String {
    let escaped = input.replace('"', "\"\"");
    format!("\"{}\"", escaped)
}
