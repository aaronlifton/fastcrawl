//! Shared pgvector table helpers reused by binaries.

use anyhow::Result;

/// Fully-qualified Postgres table name (schema + table).
#[derive(Debug, Clone)]
pub struct TableName {
    schema: String,
    table: String,
}

impl TableName {
    /// Builds a new table identifier.
    pub fn new<S, T>(schema: S, table: T) -> Result<Self>
    where
        S: Into<String>,
        T: Into<String>,
    {
        let schema = schema.into();
        let table = table.into();
        anyhow::ensure!(!schema.trim().is_empty(), "schema name is required");
        anyhow::ensure!(!table.trim().is_empty(), "table name is required");
        Ok(Self { schema, table })
    }

    /// Fully-qualified table reference with quoted identifiers.
    pub fn qualified(&self) -> String {
        format!("{}.{}", quote_ident(&self.schema), quote_ident(&self.table))
    }

    /// Returns the raw schema string.
    pub fn schema(&self) -> &str {
        &self.schema
    }

    /// Returns the raw table string.
    pub fn table(&self) -> &str {
        &self.table
    }

    /// Index name used for the generated text_tsv GIN index.
    pub fn fts_index_name(&self) -> String {
        format!(
            "{}_{}_text_tsv_idx",
            sanitize_ident(&self.schema),
            sanitize_ident(&self.table)
        )
    }
}

/// Quotes Postgres identifiers, escaping embedded quotes.
pub fn quote_ident(input: &str) -> String {
    let escaped = input.replace('"', "\"\"");
    format!("\"{}\"", escaped)
}

fn sanitize_ident(input: &str) -> String {
    input
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}
