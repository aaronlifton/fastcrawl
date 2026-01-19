use std::env;
use std::fs;
use std::io::{self, Read};
use std::process;

fn main() {
    if let Err(err) = run() {
        eprintln!("{}: {err}", env!("CARGO_PKG_NAME"));
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args();
    let program = args
        .next()
        .unwrap_or_else(|| env!("CARGO_PKG_NAME").to_string());

    let input = match args.next() {
        Some(flag) if is_help_flag(&flag) => {
            print_help(&program);
            return Ok(());
        }
        Some(flag) if is_version_flag(&flag) => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        Some(path) if path == "-" => Input::Stdin,
        Some(path) => Input::File(path),
        None => Input::Stdin,
    };

    if let Some(extra) = args.next() {
        return Err(format!("unexpected argument: {extra}\n{}", usage(&program)));
    }

    let html = match input {
        Input::Stdin => read_stdin()?,
        Input::File(path) => read_file(&path)?,
    };

    let blocks = wiki_parser::extract_blocks(&html);
    let json = serde_json::to_string_pretty(&blocks)
        .map_err(|err| format!("failed to serialize JSON: {err}"))?;
    println!("{json}");
    Ok(())
}

enum Input {
    Stdin,
    File(String),
}

fn is_help_flag(arg: &str) -> bool {
    arg == "-h" || arg == "--help"
}

fn is_version_flag(arg: &str) -> bool {
    arg == "-V" || arg == "--version"
}

fn read_file(path: &str) -> Result<String, String> {
    fs::read_to_string(path).map_err(|err| format!("failed to read '{path}': {err}"))
}

fn read_stdin() -> Result<String, String> {
    let mut buf = String::new();
    io::stdin()
        .read_to_string(&mut buf)
        .map_err(|err| format!("failed to read stdin: {err}"))?;
    Ok(buf)
}

fn print_help(program: &str) {
    println!(
        "{}\n\nOptions:\n  -h, --help      Show this message\n  -V, --version   Print package version",
        usage(program)
    );
}

fn usage(program: &str) -> String {
    format!(
        "Usage: {program} [HTML_FILE|-]\n\n\
         Provide a path to a downloaded Wikipedia HTML file or '-' to read from stdin. \
         When no argument is passed, stdin is used."
    )
}
