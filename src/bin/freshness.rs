use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use fastcrawl::ManifestRecord;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(
    name = "fastcrawl-freshness",
    about = "Detect manifest drift and emit embedding refresh plans"
)]
struct FreshnessCli {
    /// Path to the latest manifest JSONL emitted by normalization.
    #[arg(
        long,
        env = "FASTCRAWL_MANIFEST_CURRENT",
        default_value = "data/wiki_manifest.jsonl"
    )]
    current_manifest: PathBuf,

    /// Optional previous manifest snapshot for comparison.
    #[arg(long, env = "FASTCRAWL_MANIFEST_PREVIOUS")]
    previous_manifest: Option<PathBuf>,

    /// Location to write the refresh plan JSONL.
    #[arg(
        long,
        env = "FASTCRAWL_REFRESH_PLAN",
        default_value = "data/refresh_plan.jsonl"
    )]
    plan_output: PathBuf,

    /// Ledger capturing every refresh decision (append-only JSONL).
    #[arg(
        long,
        env = "FASTCRAWL_REFRESH_LEDGER",
        default_value = "data/embedding_ledger.jsonl"
    )]
    ledger_output: PathBuf,

    /// Optional shell command executed after the plan is written.
    /// The environment variable FASTCRAWL_REFRESH_PLAN resolves to the plan path.
    #[arg(long)]
    exec_plan: Option<String>,

    /// Skip writing the plan/ledger, useful for dry runs.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum RefreshReason {
    New,
    Changed,
    Deleted,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum RefreshAction {
    Embed,
    Delete,
}

#[derive(Debug, Serialize, Deserialize)]
struct RefreshPlanEntry {
    url: String,
    checksum: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_seen_epoch_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_checksum: Option<u32>,
    reason: RefreshReason,
    action: RefreshAction,
}

#[derive(Debug, Serialize)]
struct LedgerEntry {
    emitted_at_epoch_ms: u64,
    url: String,
    checksum: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_checksum: Option<u32>,
    reason: RefreshReason,
    action: RefreshAction,
}

fn main() -> Result<()> {
    let cli = FreshnessCli::parse();
    let current = load_manifest_map(&cli.current_manifest)?;
    let previous = if let Some(path) = &cli.previous_manifest {
        load_manifest_map(path)?
    } else {
        HashMap::new()
    };
    let entries = build_plan(&current, &previous);
    render_stats(&entries, current.len(), previous.len());
    if cli.dry_run {
        println!("dry run enabled; skipping plan + ledger writes");
    } else {
        write_plan(&cli.plan_output, &entries)?;
        append_ledger(&cli.ledger_output, &entries)?;
    }
    if let Some(cmd) = &cli.exec_plan {
        if cli.dry_run {
            println!("dry run enabled; skipping exec-plan");
        } else {
            run_exec(cmd, &cli.plan_output)?;
        }
    }
    Ok(())
}

fn load_manifest_map(path: &Path) -> Result<HashMap<String, ManifestRecord>> {
    let file = File::open(path).with_context(|| format!("failed to open manifest {:?}", path))?;
    let reader = BufReader::new(file);
    let mut records = HashMap::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read manifest line {}", idx + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let record: ManifestRecord = serde_json::from_str(&line)
            .with_context(|| format!("invalid manifest record at line {}", idx + 1))?;
        records.insert(record.url.clone(), record);
    }
    Ok(records)
}

fn build_plan(
    current: &HashMap<String, ManifestRecord>,
    previous: &HashMap<String, ManifestRecord>,
) -> Vec<RefreshPlanEntry> {
    let mut plan = Vec::new();
    for (url, record) in current {
        match previous.get(url) {
            None => plan.push(RefreshPlanEntry {
                url: url.clone(),
                checksum: record.checksum,
                last_seen_epoch_ms: Some(record.last_seen_epoch_ms),
                previous_checksum: None,
                reason: RefreshReason::New,
                action: RefreshAction::Embed,
            }),
            Some(prev) if prev.checksum != record.checksum => plan.push(RefreshPlanEntry {
                url: url.clone(),
                checksum: record.checksum,
                last_seen_epoch_ms: Some(record.last_seen_epoch_ms),
                previous_checksum: Some(prev.checksum),
                reason: RefreshReason::Changed,
                action: RefreshAction::Embed,
            }),
            _ => {}
        }
    }
    for (url, record) in previous {
        if !current.contains_key(url) {
            plan.push(RefreshPlanEntry {
                url: url.clone(),
                checksum: record.checksum,
                last_seen_epoch_ms: Some(record.last_seen_epoch_ms),
                previous_checksum: Some(record.checksum),
                reason: RefreshReason::Deleted,
                action: RefreshAction::Delete,
            });
        }
    }
    plan
}

fn write_plan(path: &Path, entries: &[RefreshPlanEntry]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("failed to create {:?}", parent))?;
    }
    let mut file =
        File::create(path).with_context(|| format!("failed to create plan {:?}", path))?;
    for entry in entries {
        let line = serde_json::to_string(entry)?;
        writeln!(file, "{line}")?;
    }
    println!("wrote {} refresh entries to {:?}", entries.len(), path);
    Ok(())
}

fn append_ledger(path: &Path, entries: &[RefreshPlanEntry]) -> Result<()> {
    if entries.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("failed to create {:?}", parent))?;
    }
    let timestamp = epoch_ms();
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open ledger {:?}", path))?;
    for entry in entries {
        let ledger_entry = LedgerEntry {
            emitted_at_epoch_ms: timestamp,
            url: entry.url.clone(),
            checksum: entry.checksum,
            previous_checksum: entry.previous_checksum,
            reason: entry.reason,
            action: entry.action,
        };
        let line = serde_json::to_string(&ledger_entry)?;
        writeln!(file, "{line}")?;
    }
    println!(
        "appended {} refresh decisions to ledger {:?}",
        entries.len(),
        path
    );
    Ok(())
}

fn epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|dur| dur.as_millis() as u64)
        .unwrap_or(0)
}

fn run_exec(command: &str, plan_path: &Path) -> Result<()> {
    println!("executing refresh hook: {}", command);
    let status = Command::new("sh")
        .arg("-c")
        .arg(command)
        .env("FASTCRAWL_REFRESH_PLAN", plan_path)
        .status()
        .context("failed to spawn exec-plan command")?;
    anyhow::ensure!(
        status.success(),
        "exec-plan command exited with status {status}"
    );
    Ok(())
}

fn render_stats(entries: &[RefreshPlanEntry], current_total: usize, previous_total: usize) {
    let mut new_count = 0usize;
    let mut changed_count = 0usize;
    let mut deleted_count = 0usize;
    for entry in entries {
        match entry.reason {
            RefreshReason::New => new_count += 1,
            RefreshReason::Changed => changed_count += 1,
            RefreshReason::Deleted => deleted_count += 1,
        }
    }
    println!("--- Freshness Planner ---");
    println!("current manifest URLs: {}", current_total);
    println!("previous manifest URLs: {}", previous_total);
    println!("new pages: {new_count}");
    println!("changed pages: {changed_count}");
    println!("deleted pages: {deleted_count}");
}
