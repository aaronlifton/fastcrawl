# wiki_parser

Extract readable text from Wikipedia HTML pages.

## Install

Library:

```
cargo add wiki_parser
```

CLI:

```
cargo install wiki_parser
```

## Usage

### Library

```rust
use wiki_parser::{extract_blocks, extract_text, BlockKind};

let html = r#"
<div id="mw-content-text">
  <h1>Title</h1>
  <p>Hello <sup>[1]</sup>world.</p>
  <ul><li>Item one</li><li>Item two</li></ul>
</div>
"#;
let text = extract_text(html);
assert_eq!(text, "Title\nHello world.\nItem one\nItem two");

let blocks = extract_blocks(html);
assert_eq!(blocks[0].kind, BlockKind::Heading);
```

### CLI

```
cargo run -- path/to/page.html > page.json
```

The CLI reads from the provided HTML file (or `-` for stdin; stdin is also the
default when no argument is given) and prints a JSON array of blocks, each with
`type`, `text`, `section_path`, and `index` fields. Block types are `paragraph`,
`heading`, `list_item`, and `code`. Extra arguments are rejected. This makes it easy to `curl` a
Wikipedia page, save it locally, and convert it into an LLM-friendly format.

Example:

```
curl -s https://en.wikipedia.org/wiki/Rust_(programming_language) \
  -o page.html
cargo run -- page.html > page.json
```

Options:

- `-h`, `--help` show usage.
- `-V`, `--version` print package version.

## Notes

Extraction rules:

- Root selection: first match of `#mw-content-text`, then `#bodyContent`, then
  `body`, else the root element.
- Node extraction: only `p`, `h1`-`h6`, and `li` under the root.
- Output format: one line per extracted node, joined with `\n` and with
  whitespace collapsed.
- If no `p`/`h*`/`li` nodes are found, fall back to normalized full-root text.

Ignored elements:

- Tags: `script`, `style`, `table`, `sup`, `noscript`, `figure`, `header`,
  `footer`, `nav`, `aside`.
- Classes: `ambox`, `authority-control`, `autocollapse`, `catlinks`, `cmbox`,
  `collapsible-list`, `dablink`, `fmbox`, `hatnote`, `imbox`, `infobox`,
  `infobox-above`, `infobox-caption`, `infobox-data`, `infobox-full-data`,
  `infobox-header`, `infobox-image`, `infobox-label`, `infobox-signature`,
  `infobox-subheader`, `infobox-title`, `mbox-image`, `mbox-small`,
  `mbox-text`, `metadata`, `messagebox`, `multiple-issues`, `mw-collapsible`,
  `mw-collapsible-content`, `mw-editsection`, `mw-editsection-bracket`,
  `mw-footer`, `mw-footer-container`, `mw-hidden-catlinks`,
  `mw-normal-catlinks`, `mw-portlet-sticky-header-toc`,
  `mw-references-columns`, `mw-references-wrap`, `navbox`,
  `navbox-abovebelow`, `navbox-columns-table`, `navbox-even`, `navbox-group`,
  `navbox-image`, `navbox-inner`, `navbox-list`, `navbox-list-with-group`,
  `navbox-odd`, `navbox-styles`, `navbox-subgroup`, `navbox-title`,
  `nowraplinks`, `ombox`, `pie-thumb`, `printfooter`, `refbegin`,
  `reference`, `reference-text`, `reflist`, `references`,
  `references-column-width`, `shortdescription`, `sidebar`, `sidebar-content`,
  `sidebar-navbar`, `sidebar-title`, `thumb`, `thumbcaption`, `thumbimage`,
  `thumbinner`, `toc`, `toccolours`, `vertical-navbox`,
  `vector-page-titlebar-toc`, `vector-sticky-header-toc`, plus any class
  starting with `vector-toc` or `toclimit-`.

Provide the raw HTML from a page fetcher of your choice.
