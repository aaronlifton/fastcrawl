use std::io::Write;
use std::process::{Command, Stdio};

fn expected_with_newline(expected: &str) -> String {
    format!("{}\n", expected.trim_end_matches('\n'))
}

#[test]
fn cli_reads_file_path() {
    let output = Command::new(env!("CARGO_BIN_EXE_wiki_parser"))
        .arg("tests/fixtures/html/sections-and-list.html")
        .output()
        .expect("run CLI");

    assert!(
        output.status.success(),
        "cli exited with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let expected = include_str!("fixtures/expected/sections-and-list.txt");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        expected_with_newline(expected),
    );
}

#[test]
fn cli_reads_stdin_when_no_args() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_wiki_parser"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn CLI");

    let html = include_str!("fixtures/html/lead-with-infobox.html");
    child
        .stdin
        .as_mut()
        .expect("stdin open")
        .write_all(html.as_bytes())
        .expect("write stdin");

    let output = child.wait_with_output().expect("read CLI output");
    assert!(
        output.status.success(),
        "cli exited with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let expected = include_str!("fixtures/expected/lead-with-infobox.txt");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        expected_with_newline(expected),
    );
}
