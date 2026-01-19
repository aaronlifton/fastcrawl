use pretty_assertions::assert_eq;

use wiki_parser::extract_text;

#[test]
fn fixtures_match_expected_output() {
    let cases = [
        (
            "bodycontent-fallback",
            include_str!("fixtures/html/bodycontent-fallback.html"),
            include_str!("fixtures/expected/bodycontent-fallback.txt"),
        ),
        (
            "lead-with-infobox",
            include_str!("fixtures/html/lead-with-infobox.html"),
            include_str!("fixtures/expected/lead-with-infobox.txt"),
        ),
        (
            "no-structured-nodes",
            include_str!("fixtures/html/no-structured-nodes.html"),
            include_str!("fixtures/expected/no-structured-nodes.txt"),
        ),
        (
            "references-and-navbox",
            include_str!("fixtures/html/references-and-navbox.html"),
            include_str!("fixtures/expected/references-and-navbox.txt"),
        ),
        (
            "sections-and-list",
            include_str!("fixtures/html/sections-and-list.html"),
            include_str!("fixtures/expected/sections-and-list.txt"),
        ),
        (
            "toc-hatnote-and-thumb",
            include_str!("fixtures/html/toc-hatnote-and-thumb.html"),
            include_str!("fixtures/expected/toc-hatnote-and-thumb.txt"),
        ),
        (
            "vector-toc-and-messageboxes",
            include_str!("fixtures/html/vector-toc-and-messageboxes.html"),
            include_str!("fixtures/expected/vector-toc-and-messageboxes.txt"),
        ),
    ];

    for (name, html, expected) in cases {
        let actual = extract_text(html);
        assert_eq!(
            actual,
            expected.trim_end_matches('\n'),
            "fixture mismatch: {name}"
        );
    }
}
