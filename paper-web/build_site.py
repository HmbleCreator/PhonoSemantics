from __future__ import annotations

from dataclasses import dataclass
from html import escape, unescape
from pathlib import Path
import re
import shutil
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = Path(__file__).resolve().parent
TEX_PATH = SITE_DIR / "downloads" / "phonosemantic_zenodo.tex"
BIB_PATH = SITE_DIR / "references.bib"
README_PATH = ROOT / "README.md"
OUTPUT_PATH = SITE_DIR / "index.html"
ASSETS_DIR = SITE_DIR / "assets"
DOWNLOADS_DIR = SITE_DIR / "downloads"
KNOWN_DOI = "10.5281/zenodo.19508958"
MATH_TOKEN_RE = re.compile(r"@@MATH(\d+)@@")
REF_TOKEN_RE = re.compile(r"@@REF:([^@]+)@@")


@dataclass
class Heading:
    level: int
    title: str
    number: str
    anchor: str


def remove_comments(source: str) -> str:
    cleaned: list[str] = []
    for line in source.splitlines():
        current: list[str] = []
        backslashes = 0
        for ch in line:
            if ch == "\\":
                backslashes += 1
                current.append(ch)
                continue
            if ch == "%" and backslashes % 2 == 0:
                break
            backslashes = 0
            current.append(ch)
        cleaned.append("".join(current))
    return "\n".join(cleaned)


def skip_ws(text: str, start: int) -> int:
    idx = start
    while idx < len(text) and text[idx].isspace():
        idx += 1
    return idx


def extract_balanced(text: str, start: int, open_char: str = "{", close_char: str = "}") -> tuple[str, int]:
    if start >= len(text) or text[start] != open_char:
        raise ValueError(f"Expected {open_char!r} at position {start}")

    depth = 0
    content: list[str] = []
    idx = start
    while idx < len(text):
        ch = text[idx]
        if ch == "\\" and idx + 1 < len(text):
            if depth > 0:
                content.append(ch)
                content.append(text[idx + 1])
            idx += 2
            continue
        if ch == open_char:
            depth += 1
            if depth > 1:
                content.append(ch)
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return "".join(content), idx + 1
            content.append(ch)
        else:
            if depth > 0:
                content.append(ch)
        idx += 1
    raise ValueError(f"Unbalanced {open_char!r}{close_char!r} group")


def find_command_value(source: str, command: str) -> str:
    match = re.search(rf"\\{re.escape(command)}(?![A-Za-z])\s*", source)
    if not match:
        return ""
    idx = skip_ws(source, match.end())
    if idx >= len(source) or source[idx] != "{":
        return ""
    value, _ = extract_balanced(source, idx)
    return value.strip()


def slugify(text: str) -> str:
    plain = unescape(re.sub(r"<[^>]+>", "", text))
    slug = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")
    return slug or "section"


def unique_anchor(seed: str, used: set[str]) -> str:
    base = slugify(seed)
    candidate = base
    counter = 2
    while candidate in used:
        candidate = f"{base}-{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_bibliography(path: Path) -> dict[str, dict[str, str]]:
    text = read_text(path)
    entries: dict[str, dict[str, str]] = {}
    index = 0

    while True:
        start = text.find("@", index)
        if start == -1:
            break
        brace = text.find("{", start)
        if brace == -1:
            break
        entry_type = text[start + 1:brace].strip().lower()
        body, index = extract_balanced(text, brace)
        if "," not in body:
            continue
        key, rest = body.split(",", 1)
        fields = parse_bib_fields(rest)
        fields["entry_type"] = entry_type
        entries[key.strip()] = fields
    return entries


def parse_bib_fields(source: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    idx = 0
    length = len(source)

    while idx < length:
        while idx < length and source[idx] in ", \n\r\t":
            idx += 1
        if idx >= length:
            break

        name_start = idx
        while idx < length and source[idx] not in "=\n\r\t ":
            idx += 1
        name = source[name_start:idx].strip().lower()
        while idx < length and source[idx] != "=":
            idx += 1
        if idx >= length:
            break
        idx += 1
        idx = skip_ws(source, idx)
        if idx >= length:
            break

        if source[idx] == "{":
            value, idx = extract_balanced(source, idx)
        elif source[idx] == '"':
            idx += 1
            value_start = idx
            while idx < length and source[idx] != '"':
                idx += 1
            value = source[value_start:idx]
            idx += 1
        else:
            value_start = idx
            while idx < length and source[idx] not in ",\n\r":
                idx += 1
            value = source[value_start:idx]

        fields[name] = value.strip()
    return fields


def protect_math(text: str) -> tuple[str, list[str]]:
    tokens: list[str] = []
    output: list[str] = []
    idx = 0

    while idx < len(text):
        if text.startswith(r"\(", idx):
            end = text.find(r"\)", idx + 2)
            if end != -1:
                tokens.append(text[idx + 2:end].strip())
                output.append(f"@@MATH{len(tokens) - 1}@@")
                idx = end + 2
                continue

        if text[idx] == "$" and (idx == 0 or text[idx - 1] != "\\"):
            end = idx + 1
            while end < len(text):
                if text[end] == "$" and text[end - 1] != "\\":
                    break
                end += 1
            if end < len(text):
                tokens.append(text[idx + 1:end].strip())
                output.append(f"@@MATH{len(tokens) - 1}@@")
                idx = end + 1
                continue

        output.append(text[idx])
        idx += 1

    return "".join(output), tokens


def parse_token_argument(text: str, idx: int) -> tuple[str, int]:
    idx = skip_ws(text, idx)
    if idx >= len(text):
        return "", idx
    if text[idx] == "{":
        return extract_balanced(text, idx)
    if text[idx] == "\\":
        end = idx + 1
        if end < len(text) and text[end].isalpha():
            while end < len(text) and text[end].isalpha():
                end += 1
        else:
            end += 1
        return text[idx:end], end
    return text[idx], idx + 1


def is_accent_sequence(text: str, idx: int) -> bool:
    if idx + 2 >= len(text):
        return False
    next_char = text[idx + 2]
    if next_char == "{":
        return True
    if next_char == "\\":
        return True
    if next_char.isalpha():
        return idx + 3 >= len(text) or not text[idx + 3].isalpha()
    return True


def accent_lookup(kind: str, token: str) -> str:
    if token.startswith("\\"):
        token = token[1:]
    if token == "i":
        token = "i"

    tables = {
        "=": {"a": "a\u0304", "A": "A\u0304", "i": "i\u0304", "I": "I\u0304", "u": "u\u0304", "U": "U\u0304", "r": "r\u0304", "R": "R\u0304", "l": "l\u0304", "L": "L\u0304"},
        "d": {"n": "n\u0323", "N": "N\u0323", "t": "t\u0323", "T": "T\u0323", "d": "d\u0323", "D": "D\u0323", "s": "s\u0323", "S": "S\u0323", "l": "l\u0323", "L": "L\u0323", "r": "r\u0323", "R": "R\u0323"},
        "~": {"n": "\u00f1", "N": "\u00d1", "a": "\u00e3", "A": "\u00c3"},
        "'": {"s": "\u015b", "S": "\u015a", "a": "\u00e1", "A": "\u00c1", "i": "\u00ed", "I": "\u00cd", "u": "\u00fa", "U": "\u00da", "e": "\u00e9", "E": "\u00c9", "o": "\u00f3", "O": "\u00d3"},
        "v": {"o": "\u01d2", "O": "\u01d1", "s": "\u0161", "S": "\u0160"},
        "`": {"a": "\u00e0", "A": "\u00c0", "i": "\u00ec", "I": "\u00cc", "u": "\u00f9", "U": "\u00d9"},
        '"': {"a": "\u00e4", "A": "\u00c4", "e": "\u00eb", "E": "\u00cb", "i": "\u00ef", "I": "\u00cf", "o": "\u00f6", "O": "\u00d6", "u": "\u00fc", "U": "\u00dc", "y": "\u00ff", "Y": "\u0178"},
    }

    return tables.get(kind, {}).get(token, token)


def format_authors(raw: str) -> list[str]:
    authors = [part.strip() for part in raw.split(" and ") if part.strip()]
    names: list[str] = []
    for author in authors:
        if "," in author:
            last = author.split(",", 1)[0].strip()
        else:
            tokens = author.split()
            last = tokens[-1] if tokens else author
        names.append(last)
    return names


def citation_text(key: str, bibliography: dict[str, dict[str, str]]) -> tuple[str, str]:
    entry = bibliography.get(key, {})
    author_field = render_plain(entry.get("author", key), bibliography)
    authors = format_authors(author_field)
    if not authors:
        authors_text = key
    elif len(authors) == 1:
        authors_text = authors[0]
    elif len(authors) == 2:
        authors_text = f"{authors[0]} and {authors[1]}"
    else:
        authors_text = f"{authors[0]} et al."

    year = render_plain(entry.get("year", "n.d."), bibliography)
    return authors_text, year


def render_citation(kind: str, keys_text: str, bibliography: dict[str, dict[str, str]]) -> str:
    keys = [key.strip() for key in keys_text.split(",") if key.strip()]
    if not keys:
        return ""

    if kind == "citeauthor":
        parts = []
        for key in keys:
            authors, _ = citation_text(key, bibliography)
            parts.append(f'<a class="citation" href="#ref-{escape(key)}">{escape(authors)}</a>')
        return ", ".join(parts)

    if kind == "citet":
        if len(keys) == 1:
            key = keys[0]
            authors, year = citation_text(key, bibliography)
            return f'<a class="citation" href="#ref-{escape(key)}">{escape(authors)} ({escape(year)})</a>'
        rendered = []
        for key in keys:
            authors, year = citation_text(key, bibliography)
            rendered.append(f'<a class="citation" href="#ref-{escape(key)}">{escape(authors)} ({escape(year)})</a>')
        return "; ".join(rendered)

    rendered = []
    for key in keys:
        authors, year = citation_text(key, bibliography)
        rendered.append(f'<a class="citation" href="#ref-{escape(key)}">{escape(authors)}, {escape(year)}</a>')
    return "(" + "; ".join(rendered) + ")"


def render_non_math(text: str, bibliography: dict[str, dict[str, str]]) -> str:
    output: list[str] = []
    idx = 0

    while idx < len(text):
        if text.startswith(r"\\", idx):
            idx += 2
            if idx < len(text) and text[idx] == "[":
                _, idx = extract_balanced(text, idx, "[", "]")
            output.append("<br>")
            continue

        ch = text[idx]
        if ch == "{":
            value, idx = extract_balanced(text, idx)
            output.append(render_inline(value, bibliography))
            continue
        if ch == "}":
            idx += 1
            continue
        if ch == "~":
            output.append(" ")
            idx += 1
            continue
        if ch == "&":
            output.append("&amp;")
            idx += 1
            continue
        if ch == "<":
            output.append("&lt;")
            idx += 1
            continue
        if ch == ">":
            output.append("&gt;")
            idx += 1
            continue
        if ch != "\\":
            output.append(ch)
            idx += 1
            continue

        if idx + 1 < len(text) and text[idx + 1] in "%&_#${}":
            output.append(escape(text[idx + 1]))
            idx += 2
            continue
        if idx + 1 < len(text) and text[idx + 1] == " ":
            output.append(" ")
            idx += 2
            continue
        if idx + 1 < len(text) and text[idx + 1] in "=d'v`~\"" and is_accent_sequence(text, idx):
            accent = text[idx + 1]
            token, idx = parse_token_argument(text, idx + 2)
            output.append(escape(accent_lookup(accent, token)))
            continue

        cmd_start = idx + 1
        if cmd_start < len(text) and text[cmd_start].isalpha():
            cmd_end = cmd_start
            while cmd_end < len(text) and text[cmd_end].isalpha():
                cmd_end += 1
        else:
            cmd_end = min(cmd_start + 1, len(text))
        command = text[cmd_start:cmd_end]
        idx = cmd_end

        if command in {"small", "normalsize", "large", "centering", "noindent", "smallskip", "medskip"}:
            continue
        if command == "textcopyright":
            output.append("&copy;")
            continue
        if command in {"textbf", "textit", "emph", "texttt"}:
            argument, idx = parse_token_argument(text, idx)
            tag = {"textbf": "strong", "textit": "em", "emph": "em", "texttt": "code"}[command]
            output.append(f"<{tag}>{render_inline(argument, bibliography)}</{tag}>")
            continue
        if command == "href":
            url, idx = parse_token_argument(text, idx)
            label, idx = parse_token_argument(text, idx)
            safe_url = escape(render_plain(url, bibliography))
            output.append(f'<a href="{safe_url}" target="_blank" rel="noreferrer">{render_inline(label, bibliography)}</a>')
            continue
        if command == "url":
            url, idx = parse_token_argument(text, idx)
            safe_url = escape(render_plain(url, bibliography))
            output.append(f'<a href="{safe_url}" target="_blank" rel="noreferrer">{safe_url}</a>')
            continue
        if command in {"citep", "citet", "citeauthor"}:
            argument, idx = parse_token_argument(text, idx)
            output.append(render_citation(command, argument, bibliography))
            continue
        if command == "ref":
            argument, idx = parse_token_argument(text, idx)
            output.append(f"@@REF:{argument}@@")
            continue
        if command == "i":
            output.append("i")
            continue
        if command == "LaTeX":
            output.append("LaTeX")
            continue
        if command == "TeX":
            output.append("TeX")
            continue
        if command in {",", ";", ":", "!"}:
            output.append(" ")
            continue

        output.append(escape("\\" + command))

    rendered = "".join(output)
    rendered = rendered.replace("---", "&mdash;").replace("--", "&ndash;")
    rendered = rendered.replace("``", "&ldquo;").replace("''", "&rdquo;")
    rendered = rendered.replace("~", " ")
    rendered = re.sub(r"\s+\n", "\n", rendered)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return rendered


def render_inline(text: str, bibliography: dict[str, dict[str, str]]) -> str:
    protected, math_tokens = protect_math(text)
    rendered = render_non_math(protected, bibliography)

    def math_replacer(match: re.Match[str]) -> str:
        token_index = int(match.group(1))
        if token_index >= len(math_tokens):
            return match.group(0)
        math_source = normalize_math_source(math_tokens[token_index])
        return f'<span class="math-inline">\\({escape(math_source)}\\)</span>'

    return MATH_TOKEN_RE.sub(math_replacer, rendered)


def render_plain(text: str, bibliography: dict[str, dict[str, str]]) -> str:
    html_value = render_inline(text, bibliography)
    plain = re.sub(r"<[^>]+>", "", html_value)
    plain = REF_TOKEN_RE.sub("?", plain)
    return unescape(plain).strip()


def strip_html_markup(text: str) -> str:
    plain = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    plain = re.sub(r"<[^>]+>", "", plain)
    plain = re.sub(r"\s+", " ", unescape(plain))
    return plain.strip()


def normalize_math_source(text: str) -> str:
    result: list[str] = []
    idx = 0

    while idx < len(text):
        if text[idx] != "\\":
            result.append(text[idx])
            idx += 1
            continue

        if idx + 1 < len(text) and text[idx + 1] in "{}":
            result.append(text[idx + 1])
            idx += 2
            continue

        if idx + 1 < len(text) and text[idx + 1] in "=d'v`~\"" and is_accent_sequence(text, idx):
            accent = text[idx + 1]
            token, idx = parse_token_argument(text, idx + 2)
            result.append(accent_lookup(accent, token))
            continue

        result.append(text[idx])
        idx += 1

    return "".join(result)


def resolve_references(text: str, labels: dict[str, str]) -> str:
    def replacer(match: re.Match[str]) -> str:
        key = match.group(1)
        return escape(labels.get(key, "??"))

    return REF_TOKEN_RE.sub(replacer, text)


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [part.strip() for part in parts if part.strip()]


def find_label(line: str) -> str | None:
    match = re.search(r"\\label\{([^}]+)\}", line)
    return match.group(1).strip() if match else None


def discover_doi(tex_source: str) -> str:
    candidates = re.findall(r"10\.5281/zenodo\.(?:\d+|XXXXXXX)", tex_source)
    for candidate in candidates:
        if "XXXXXXX" not in candidate:
            return candidate
    if README_PATH.exists():
        readme = read_text(README_PATH)
        readme_match = re.search(r"10\.5281/zenodo\.\d+", readme)
        if readme_match:
            return readme_match.group(0)
    return KNOWN_DOI


def collect_environment(lines: list[str], start: int, name: str) -> tuple[str, int]:
    begin = rf"\begin{{{name}}}"
    end = rf"\end{{{name}}}"
    depth = 0
    collected: list[str] = []
    idx = start

    while idx < len(lines):
        line = lines[idx]
        if begin in line:
            depth += 1
            if depth == 1:
                idx += 1
                continue
        if end in line:
            depth -= 1
            if depth == 0:
                return "\n".join(collected).strip(), idx + 1
        if depth >= 1:
            collected.append(line)
        idx += 1
    raise ValueError(f"Environment {name!r} was not closed")


def collect_bracket_math(lines: list[str], start: int) -> tuple[str, int]:
    collected: list[str] = []
    idx = start + 1
    while idx < len(lines):
        if lines[idx].strip() == r"\]":
            return "\n".join(collected).strip(), idx + 1
        collected.append(lines[idx])
        idx += 1
    raise ValueError("Display math block was not closed")


def resolve_asset_path(include_target: str) -> tuple[str, Path | None]:
    raw = include_target.strip()
    source = (ROOT / raw).resolve()
    if source.exists():
        return f"assets/{source.name}", source

    stem = Path(raw).stem
    for extension in (".png", ".jpg", ".jpeg", ".webp", ".svg"):
        candidate = (ROOT / f"{stem}{extension}").resolve()
        if candidate.exists():
            return f"assets/{candidate.name}", candidate
    return "", None


def parse_tabular(tabular_source: str, bibliography: dict[str, dict[str, str]]) -> tuple[list[list[str]], list[list[str]]]:
    lines = [line.strip() for line in tabular_source.splitlines() if line.strip()]
    header_rows: list[list[str]] = []
    body_rows: list[list[str]] = []
    in_header = True

    for line in lines:
        if line.startswith(r"\begin{tabular}") or line.startswith(r"\end{tabular}"):
            continue
        if line.startswith(r"\toprule"):
            continue
        if line.startswith(r"\midrule"):
            in_header = False
            continue
        if line.startswith(r"\bottomrule"):
            continue
        if line.startswith(r"\renewcommand") or line.startswith(r"\arraystretch") or line == r"\centering":
            continue

        row_text = line
        while row_text.endswith("\\"):
            row_text = row_text[:-2].rstrip()

        cells = [render_inline(cell.strip(), bibliography) for cell in row_text.split("&")]
        if not any(unescape(re.sub(r"<[^>]+>", "", cell)).strip() for cell in cells):
            continue
        if in_header:
            header_rows.append(cells)
        else:
            body_rows.append(cells)

    return header_rows, body_rows


def extract_tabular_source(source: str) -> tuple[str, str]:
    begin_marker = r"\begin{tabular}"
    start = source.find(begin_marker)
    if start == -1:
        return "", ""

    idx = start + len(begin_marker)
    idx = skip_ws(source, idx)
    column_spec = ""
    if idx < len(source) and source[idx] == "{":
        column_spec, idx = extract_balanced(source, idx)

    end_marker = r"\end{tabular}"
    end = source.find(end_marker, idx)
    if end == -1:
        return column_spec, source[idx:]

    return column_spec, source[idx:end]


def render_tabular_card(tabular_source: str, bibliography: dict[str, dict[str, str]], caption: str = "") -> str:
    headers, rows = parse_tabular(tabular_source, bibliography)

    header_html = ""
    if headers:
        header_html += "<thead>"
        for header_row in headers:
            cells = "".join(f"<th>{cell}</th>" for cell in header_row)
            header_html += f"<tr>{cells}</tr>"
        header_html += "</thead>"

    body_html = ""
    if rows:
        body_html += "<tbody>"
        for row in rows:
            cells = "".join(f"<td>{cell}</td>" for cell in row)
            body_html += f"<tr>{cells}</tr>"
        body_html += "</tbody>"

    caption_html = ""
    if caption:
        caption_html = f"<figcaption>{caption}</figcaption>"

    return '<figure class="table-card"><div class="table-scroll"><table>' + header_html + body_html + "</table></div>" + caption_html + "</figure>"


def render_table(table_source: str, bibliography: dict[str, dict[str, str]], labels: dict[str, str], table_index: int) -> str:
    caption = find_command_value(table_source, "caption")
    label = find_label(table_source)
    _, tabular_source = extract_tabular_source(table_source)

    if label:
        labels[label] = str(table_index)

    caption_html = ""
    if caption:
        caption_html = f'<span class="caption-label">Table {table_index}.</span> {render_inline(caption, bibliography)}'

    return render_tabular_card(tabular_source, bibliography, caption_html)


def render_figure(figure_source: str, bibliography: dict[str, dict[str, str]], labels: dict[str, str], figure_index: int) -> str:
    caption = find_command_value(figure_source, "caption")
    label = find_label(figure_source)
    include_match = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", figure_source)
    image_src = ""
    image_source_path: Path | None = None
    if include_match:
        image_src, image_source_path = resolve_asset_path(include_match.group(1))
    if label:
        labels[label] = str(figure_index)

    if image_src:
        image_html = f'<img src="{escape(image_src)}" alt="{escape(render_plain(caption, bibliography) or "Figure")}">'
    else:
        image_html = '<div class="missing-asset">Figure asset could not be resolved.</div>'

    figure_html = '<figure class="paper-figure">' + image_html
    if caption:
        figure_html += f'<figcaption><span class="caption-label">Figure {figure_index}.</span> {render_inline(caption, bibliography)}</figcaption>'
    figure_html += "</figure>"

    if image_source_path:
        copy_into_site(image_source_path, ASSETS_DIR / image_source_path.name)

    return figure_html


def render_list(list_source: str, bibliography: dict[str, dict[str, str]], ordered: bool) -> str:
    items: list[str] = []
    current: list[str] = []
    for raw_line in list_source.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(r"\item"):
            if current:
                items.append(" ".join(current).strip())
            current = [line[5:].strip()]
        else:
            current.append(line)
    if current:
        items.append(" ".join(current).strip())

    tag = "ol" if ordered else "ul"
    rendered_items = "".join(f"<li>{render_inline(item, bibliography)}</li>" for item in items)
    return f'<{tag} class="paper-list">{rendered_items}</{tag}>'


def render_math_block(math_source: str) -> str:
    return f'<div class="math-display">\\[{escape(normalize_math_source(math_source.strip()))}\\]</div>'


def extract_metadata(preamble: str, bibliography: dict[str, dict[str, str]]) -> dict[str, str]:
    title = render_inline(find_command_value(preamble, "title"), bibliography)
    title = re.sub(r"^<strong>(.*)</strong>$", r"\1", title, flags=re.S).strip()
    author = render_inline(find_command_value(preamble, "author"), bibliography)
    date = render_inline(find_command_value(preamble, "date"), bibliography)
    doi = discover_doi(preamble)
    if not render_plain(title, bibliography):
        title = "Phonosemantic Grounding: Sanskrit as a Formalized Case of Motivated Sign Structure for Interpretable AI"
    date = date.replace("10.5281/zenodo.XXXXXXX", doi)
    date = date.replace("https://doi.org/10.5281/zenodo.XXXXXXX", f"https://doi.org/{doi}")
    return {
        "title": title,
        "author": author,
        "date": date,
        "doi": doi,
        "doi_url": f"https://doi.org/{doi}",
    }


def copy_into_site(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    shutil.copy2(source, target)


def build_download_bundle() -> None:
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    if TEX_PATH.exists():
        copy_into_site(TEX_PATH, DOWNLOADS_DIR / TEX_PATH.name)
    pdf_path = ROOT / "phonosemantic_zenodo_final.pdf"
    if pdf_path.exists():
        copy_into_site(pdf_path, DOWNLOADS_DIR / pdf_path.name)
    figure_path = ROOT / "phonosemantic_figure.png"
    if figure_path.exists():
        copy_into_site(figure_path, ASSETS_DIR / figure_path.name)


def build_site() -> None:
    source = remove_comments(read_text(TEX_PATH))
    source = source.replace("\nef{", r"\ref{")
    bibliography = parse_bibliography(BIB_PATH)
    preamble, body = source.split(r"\begin{document}", 1)
    body = body.split(r"\end{document}", 1)[0]
    metadata = extract_metadata(preamble, bibliography)
    lines = body.splitlines()

    labels: dict[str, str] = {}
    headings: list[Heading] = []
    used_anchors: set[str] = {"abstract", "references"}
    fragments: list[str] = []

    section_no = 0
    subsection_no = 0
    subsubsection_no = 0
    figure_no = 0
    table_no = 0
    pending_label_value: str | None = None
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines, pending_label_value
        if not paragraph_lines:
            return
        paragraph = " ".join(line.strip() for line in paragraph_lines if line.strip())
        if paragraph:
            fragments.append(f"<p>{render_inline(paragraph, bibliography)}</p>")
        paragraph_lines = []
        pending_label_value = None

    idx = 0
    while idx < len(lines):
        raw_line = lines[idx]
        line = raw_line.strip()

        if not line:
            flush_paragraph()
            idx += 1
            continue
        if line == r"\maketitle":
            idx += 1
            continue
        if line.startswith(r"\bibliographystyle") or line.startswith(r"\bibliography"):
            flush_paragraph()
            idx += 1
            continue

        label = find_label(line)
        if line.startswith(r"\label{") and pending_label_value:
            labels[label] = pending_label_value
            idx += 1
            continue

        heading_match = re.match(r"^\\(section|subsection|subsubsection)(\*)?\{", line)
        if heading_match:
            flush_paragraph()
            level_name = heading_match.group(1)
            starred = bool(heading_match.group(2))
            title, _ = extract_balanced(line, line.find("{"))
            rendered_title = render_inline(title, bibliography)

            if level_name == "section":
                if not starred:
                    section_no += 1
                    subsection_no = 0
                    subsubsection_no = 0
                    number = str(section_no)
                else:
                    number = ""
            elif level_name == "subsection":
                if not starred:
                    subsection_no += 1
                    subsubsection_no = 0
                    number = f"{section_no}.{subsection_no}"
                else:
                    number = ""
            else:
                if not starred:
                    subsubsection_no += 1
                    number = f"{section_no}.{subsection_no}.{subsubsection_no}"
                else:
                    number = ""

            anchor_seed = f"{number} {render_plain(title, bibliography)}" if number else render_plain(title, bibliography)
            anchor = unique_anchor(anchor_seed, used_anchors)
            level_map = {"section": 2, "subsection": 3, "subsubsection": 4}
            headings.append(Heading(level=level_map[level_name], title=render_plain(title, bibliography), number=number, anchor=anchor))

            number_html = f'<span class="heading-number">{escape(number)}</span>' if number else '<span class="heading-number heading-number-plain">End Matter</span>'
            tag = {2: "h2", 3: "h3", 4: "h4"}[level_map[level_name]]
            fragments.append(f'<{tag} id="{escape(anchor)}" class="paper-heading level-{level_map[level_name]}">{number_html}<span>{rendered_title}</span></{tag}>')
            pending_label_value = number
            idx += 1
            continue

        if line.startswith(r"\begin{center}"):
            flush_paragraph()
            center_source, idx = collect_environment(lines, idx, "center")
            if r"\begin{tabular}" in center_source:
                _, tabular_source = extract_tabular_source(center_source)
                fragments.append(render_tabular_card(tabular_source, bibliography))
            else:
                note_html = "".join(f"<p>{render_inline(part, bibliography)}</p>" for part in split_paragraphs(center_source))
                fragments.append(f'<section class="paper-note">{note_html}</section>')
            pending_label_value = None
            continue

        if line.startswith(r"\begin{abstract}"):
            flush_paragraph()
            abstract_source, idx = collect_environment(lines, idx, "abstract")
            keywords = ""
            keyword_match = re.search(r"\\textbf\{Keywords:\}(.*)$", abstract_source, re.S)
            if keyword_match:
                keywords = keyword_match.group(1).strip()
                abstract_source = abstract_source[:keyword_match.start()].strip()
            abstract_parts = split_paragraphs(abstract_source)
            abstract_html = "".join(f"<p>{render_inline(part, bibliography)}</p>" for part in abstract_parts)
            keyword_html = ""
            if keywords:
                keyword_items = [item.strip() for item in keywords.split(",") if item.strip()]
                tags = "".join(f"<li>{render_inline(item, bibliography)}</li>" for item in keyword_items)
                keyword_html = f'<div class="keyword-strip"><span class="keyword-label">Keywords</span><ul>{tags}</ul></div>'
            fragments.append('<section id="abstract" class="abstract-card"><div class="card-eyebrow">Abstract</div>' + abstract_html + keyword_html + "</section>")
            continue

        if line.startswith(r"\begin{figure}"):
            flush_paragraph()
            figure_source, idx = collect_environment(lines, idx, "figure")
            figure_no += 1
            fragments.append(render_figure(figure_source, bibliography, labels, figure_no))
            pending_label_value = None
            continue

        if line.startswith(r"\begin{table}"):
            flush_paragraph()
            table_source, idx = collect_environment(lines, idx, "table")
            table_no += 1
            fragments.append(render_table(table_source, bibliography, labels, table_no))
            pending_label_value = None
            continue

        if line.startswith(r"\begin{itemize}"):
            flush_paragraph()
            list_source, idx = collect_environment(lines, idx, "itemize")
            fragments.append(render_list(list_source, bibliography, ordered=False))
            pending_label_value = None
            continue

        if line.startswith(r"\begin{enumerate}"):
            flush_paragraph()
            list_source, idx = collect_environment(lines, idx, "enumerate")
            fragments.append(render_list(list_source, bibliography, ordered=True))
            pending_label_value = None
            continue

        if line.startswith(r"\begin{tabular}"):
            flush_paragraph()
            tabular_source, idx = collect_environment(lines, idx, "tabular")
            fragments.append(render_tabular_card(tabular_source, bibliography))
            pending_label_value = None
            continue

        if line.startswith(r"\begin{align}"):
            flush_paragraph()
            math_source, idx = collect_environment(lines, idx, "align")
            fragments.append(render_math_block(r"\begin{align}" + "\n" + math_source + "\n" + r"\end{align}"))
            pending_label_value = None
            continue

        if line.startswith(r"\begin{equation}"):
            flush_paragraph()
            math_source, idx = collect_environment(lines, idx, "equation")
            fragments.append(render_math_block(r"\begin{equation}" + "\n" + math_source + "\n" + r"\end{equation}"))
            pending_label_value = None
            continue

        if line == r"\[":
            flush_paragraph()
            math_source, idx = collect_bracket_math(lines, idx)
            fragments.append(render_math_block(math_source))
            pending_label_value = None
            continue

        if line in {r"\vspace{1em}", r"\smallskip"}:
            flush_paragraph()
            idx += 1
            continue

        paragraph_lines.append(line)
        idx += 1

    flush_paragraph()

    references_html = render_references(bibliography)
    headings.append(Heading(level=2, title="References", number="", anchor="references"))
    fragments.append(references_html)

    article_html = resolve_references("\n".join(fragments), labels)
    article_html = article_html.replace("10.5281/zenodo.XXXXXXX", metadata["doi"])
    article_html = article_html.replace("https://doi.org/10.5281/zenodo.XXXXXXX", metadata["doi_url"])
    nav_html = build_navigation(headings)
    page_html = render_page(metadata, article_html, nav_html)

    build_download_bundle()
    OUTPUT_PATH.write_text(unicodedata.normalize("NFC", page_html), encoding="utf-8")


def render_references(bibliography: dict[str, dict[str, str]]) -> str:
    items: list[str] = []
    for key, entry in bibliography.items():
        authors = render_plain(entry.get("author", ""), bibliography)
        year = render_plain(entry.get("year", ""), bibliography)
        title = render_inline(entry.get("title", ""), bibliography)
        journal = render_plain(entry.get("journal", ""), bibliography)
        publisher = render_plain(entry.get("publisher", ""), bibliography)
        address = render_plain(entry.get("address", ""), bibliography)
        volume = render_plain(entry.get("volume", ""), bibliography)
        number = render_plain(entry.get("number", ""), bibliography)
        pages = render_plain(entry.get("pages", ""), bibliography)
        note = render_inline(entry.get("note", ""), bibliography)

        detail_parts: list[str] = []
        if journal:
            detail = journal
            if volume:
                detail += f", {volume}"
            if number:
                detail += f"({number})"
            if pages:
                detail += f", {pages}"
            detail_parts.append(detail)
        if publisher:
            published = publisher
            if address:
                published += f", {address}"
            detail_parts.append(published)
        elif address:
            detail_parts.append(address)
        if note:
            detail_parts.append(re.sub(r"^<p>|</p>$", "", note))

        details = " ".join(part.rstrip(".") + "." for part in detail_parts if part)
        items.append(f'<li class="reference-item" id="ref-{escape(key)}"><span class="reference-authors">{escape(authors)}</span> <span class="reference-year">({escape(year)}).</span> <span class="reference-title">{title}.</span> <span class="reference-meta">{details}</span></li>')

    return '<section id="references" class="references-block"><div class="card-eyebrow">References</div><ol class="reference-list">' + "".join(items) + "</ol></section>"


def build_navigation(headings: list[Heading]) -> str:
    items = []
    for heading in headings:
        label = f"{heading.number} {heading.title}".strip()
        items.append(f'<a class="nav-link level-{heading.level}" href="#{escape(heading.anchor)}" data-target="{escape(heading.anchor)}"><span>{escape(label)}</span></a>')
    return "".join(items)


def render_page(metadata: dict[str, str], article_html: str, nav_html: str) -> str:
    title_plain = strip_html_markup(metadata["title"])
    word_count = len(re.findall(r"\w+", re.sub(r"<[^>]+>", " ", article_html)))
    read_minutes = max(1, round(word_count / 225))

    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="color-scheme" content="light dark">
  <title>__PAGE_TITLE__</title>
  <style>
    :root { color-scheme: light dark; --paper: #f4efe4; --paper-strong: #efe7d7; --panel: rgba(255,252,247,.82); --panel-strong: rgba(255,248,239,.94); --ink: #1e1c18; --muted: #645d52; --faint: #8b8377; --accent: #8b4d22; --accent-strong: #6b3210; --accent-soft: rgba(139,77,34,.12); --border: rgba(95,80,59,.16); --shadow: 0 24px 70px rgba(63,46,21,.12); --code: #f1e8d7; --serif: "Iowan Old Style","Palatino Linotype","Book Antiqua",Georgia,serif; --sans: "Aptos","Segoe UI","Helvetica Neue",sans-serif; --mono: "Cascadia Code","SFMono-Regular",Consolas,monospace; --sidebar-width: min(24rem,34vw); --content-width: 58rem; --header-offset: 1.25rem; }
    :root[data-theme="dark"] { --paper: #0f1115; --paper-strong: #161920; --panel: rgba(19,23,30,.82); --panel-strong: rgba(21,26,35,.94); --ink: #ecdfca; --muted: #b3a894; --faint: #887d6b; --accent: #d5a369; --accent-strong: #efbd84; --accent-soft: rgba(213,163,105,.14); --border: rgba(214,184,141,.14); --shadow: 0 24px 70px rgba(0,0,0,.35); --code: #1d2330; }
    * { box-sizing: border-box; } html { scroll-behavior: smooth; }
    body { margin: 0; min-height: 100vh; font-family: var(--serif); color: var(--ink); background: radial-gradient(circle at top right, rgba(196,150,86,.12), transparent 28rem), radial-gradient(circle at bottom left, rgba(120,72,30,.08), transparent 24rem), linear-gradient(180deg, var(--paper), var(--paper-strong)); transition: background .25s ease, color .25s ease; }
    body::before { content: ""; position: fixed; inset: 0; pointer-events: none; background-image: linear-gradient(rgba(120,90,54,.035) 1px, transparent 1px), linear-gradient(90deg, rgba(120,90,54,.03) 1px, transparent 1px); background-size: 100% 3rem, 3rem 100%; opacity: .35; mix-blend-mode: multiply; }
    a { color: var(--accent-strong); } a:hover { color: var(--accent); }
    .mobile-toggle { position: fixed; top: 1rem; left: 1rem; z-index: 30; display: none; border: 1px solid var(--border); background: var(--panel-strong); color: var(--ink); padding: .7rem .95rem; border-radius: 999px; backdrop-filter: blur(14px); box-shadow: var(--shadow); font: 600 .8rem/1 var(--sans); letter-spacing: .04em; text-transform: uppercase; }
    .page-shell { display: grid; grid-template-columns: minmax(18rem, var(--sidebar-width)) 1fr; min-height: 100vh; position: relative; z-index: 1; }
    .sidebar { position: sticky; top: 0; height: 100vh; padding: 1.6rem 1.15rem 1.2rem; border-right: 1px solid var(--border); background: linear-gradient(180deg, var(--panel-strong), var(--panel)); backdrop-filter: blur(18px); box-shadow: inset -1px 0 0 rgba(255,255,255,.08); overflow: auto; }
    .sidebar-top { padding-bottom: 1rem; margin-bottom: 1rem; border-bottom: 1px solid var(--border); }
    .eyebrow { margin: 0 0 .6rem; font: 700 .72rem/1 var(--sans); letter-spacing: .18em; text-transform: uppercase; color: var(--accent-strong); }
    .sidebar-title { margin: 0; font-size: 1.08rem; line-height: 1.35; font-weight: 600; }
    .sidebar-meta { margin-top: .6rem; color: var(--muted); font: .92rem/1.6 var(--sans); }
    .quick-links, .hero-links { display: flex; flex-wrap: wrap; gap: .6rem; }
    .quick-links { margin-top: 1rem; }
    .quick-links a, .hero-links a { display: inline-flex; align-items: center; gap: .35rem; padding: .42rem .75rem; border-radius: 999px; border: 1px solid var(--border); background: rgba(255,255,255,.22); color: var(--ink); text-decoration: none; font: 700 .74rem/1 var(--sans); letter-spacing: .04em; text-transform: uppercase; }
    .quick-links a:hover, .hero-links a:hover { background: var(--accent-soft); border-color: rgba(139,77,34,.28); }
    .theme-panel { margin: 1rem 0 1.1rem; padding: .9rem; border-radius: 1rem; border: 1px solid var(--border); background: rgba(255,255,255,.16); }
    .theme-panel strong { display: block; margin-bottom: .55rem; font: 600 .82rem/1.2 var(--sans); letter-spacing: .05em; text-transform: uppercase; color: var(--muted); }
    .theme-buttons { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: .4rem; }
    .theme-button { border: 1px solid var(--border); background: transparent; color: var(--ink); border-radius: 999px; padding: .55rem .35rem; font: 600 .78rem/1 var(--sans); cursor: pointer; }
    .theme-button.is-active { background: var(--accent); border-color: var(--accent); color: #fff7ee; }
    .theme-help { margin: .55rem 0 0; color: var(--faint); font: .8rem/1.45 var(--sans); }
    .nav-tree { display: flex; flex-direction: column; gap: .15rem; }
    .nav-link { display: block; color: var(--muted); text-decoration: none; padding: .45rem .75rem; border-left: 2px solid transparent; border-radius: 0 .8rem .8rem 0; transition: background .2s ease, color .2s ease, border-color .2s ease, transform .2s ease; font-family: var(--sans); line-height: 1.45; }
    .nav-link:hover { background: var(--accent-soft); color: var(--ink); transform: translateX(2px); }
    .nav-link.is-active { background: var(--accent-soft); color: var(--accent-strong); border-left-color: var(--accent-strong); font-weight: 700; }
    .nav-link.level-2 { font-size: .84rem; } .nav-link.level-3 { font-size: .78rem; padding-left: 1.35rem; } .nav-link.level-4 { font-size: .74rem; padding-left: 2rem; color: var(--faint); }
    .main { padding: 2.4rem clamp(1.2rem,3vw,2.6rem) 4rem; }
    .article { width: min(100%, var(--content-width)); margin: 0 auto; }
    .hero { padding: 2rem clamp(1.2rem,3vw,2.2rem); border: 1px solid var(--border); border-radius: 1.7rem; background: linear-gradient(140deg, rgba(255,255,255,.68), rgba(255,255,255,.36)), linear-gradient(180deg, rgba(255,248,240,.38), rgba(255,248,240,.06)); box-shadow: var(--shadow); backdrop-filter: blur(16px); }
    :root[data-theme="dark"] .hero { background: linear-gradient(140deg, rgba(36,42,55,.75), rgba(17,20,27,.62)), linear-gradient(180deg, rgba(213,163,105,.05), rgba(213,163,105,.02)); }
    .hero h1 { margin: 0; font-size: clamp(2rem,5vw,3.45rem); line-height: 1.05; letter-spacing: -.03em; }
    .hero-meta { margin-top: 1.15rem; display: flex; flex-wrap: wrap; gap: .65rem; font-family: var(--sans); color: var(--muted); }
    .hero-pill { display: inline-flex; align-items: center; gap: .35rem; padding: .42rem .72rem; border-radius: 999px; border: 1px solid var(--border); background: rgba(255,255,255,.18); font-size: .82rem; }
    .hero-links { margin-top: 1rem; font-family: var(--sans); }
    .paper-note, .abstract-card, .references-block { margin-top: 1.35rem; padding: 1.2rem 1.25rem; border: 1px solid var(--border); border-radius: 1.2rem; background: rgba(255,255,255,.14); }
    .card-eyebrow { margin: 0 0 .85rem; font: 700 .75rem/1 var(--sans); letter-spacing: .14em; text-transform: uppercase; color: var(--accent-strong); }
    .keyword-strip { margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border); }
    .keyword-label { display: inline-block; margin-bottom: .55rem; font: 700 .72rem/1 var(--sans); letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
    .keyword-strip ul { margin: 0; padding: 0; list-style: none; display: flex; flex-wrap: wrap; gap: .45rem; }
    .keyword-strip li { padding: .4rem .65rem; border-radius: 999px; border: 1px solid var(--border); background: var(--accent-soft); font: .82rem/1.2 var(--sans); }
    .paper-body { margin-top: 1.9rem; padding: 0 clamp(.15rem,1vw,.6rem); }
    .paper-heading { display: flex; align-items: baseline; gap: .75rem; margin: 2.7rem 0 1rem; scroll-margin-top: calc(var(--header-offset) + 1rem); }
    .paper-heading.level-2 { padding-top: 1.5rem; border-top: 1px solid var(--border); font-size: clamp(1.55rem,3vw,2.15rem); line-height: 1.15; }
    .paper-heading.level-3 { margin-top: 2rem; font-size: 1.32rem; line-height: 1.25; }
    .paper-heading.level-4 { margin-top: 1.55rem; font-size: 1.06rem; line-height: 1.35; font-family: var(--sans); }
    .heading-number { flex: 0 0 auto; min-width: 2.25rem; color: var(--accent-strong); font: 700 .84rem/1 var(--mono); letter-spacing: .06em; text-transform: uppercase; padding-top: .18rem; }
    .heading-number-plain { min-width: 5.8rem; color: var(--faint); }
    p { margin: 0 0 1rem; font-size: 1.04rem; line-height: 1.86; color: var(--ink); }
    .paper-note p, .abstract-card p, .references-block p { color: var(--muted); }
    strong { color: var(--ink); } em { font-style: italic; }
    code { font-family: var(--mono); background: var(--code); color: var(--accent-strong); padding: .12rem .35rem; border-radius: .3rem; font-size: .92em; }
    .paper-list { margin: 0 0 1.2rem 0; padding-left: 1.35rem; } .paper-list li { margin-bottom: .55rem; line-height: 1.8; }
    .paper-figure, .table-card { margin: 1.6rem 0 1.8rem; padding: 1rem; border-radius: 1.25rem; border: 1px solid var(--border); background: rgba(255,255,255,.12); box-shadow: 0 14px 36px rgba(52,37,19,.08); }
    .paper-figure img { display: block; width: 100%; height: auto; border-radius: .95rem; border: 1px solid var(--border); background: rgba(255,255,255,.5); }
    .paper-figure figcaption, .table-card figcaption { margin-top: .9rem; color: var(--muted); font: .92rem/1.7 var(--sans); }
    .caption-label { color: var(--accent-strong); font-weight: 700; margin-right: .25rem; }
    .table-scroll { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; min-width: 36rem; font: .97rem/1.55 var(--sans); }
    th, td { padding: .75rem .85rem; vertical-align: top; border-bottom: 1px solid var(--border); text-align: left; }
    th { color: var(--accent-strong); font-weight: 700; background: rgba(255,255,255,.12); }
    .math-inline { font-family: var(--serif); white-space: nowrap; }
    .math-display { margin: 1.3rem 0 1.5rem; padding: 1rem 1.15rem; border-left: 3px solid var(--accent); background: var(--accent-soft); border-radius: .85rem; overflow-x: auto; font-size: 1rem; }
    .citation { color: var(--accent-strong); text-decoration: none; font-family: var(--sans); font-size: .92em; } .citation:hover { text-decoration: underline; }
    .references-block { margin: 2.6rem 0 0; } .reference-list { margin: 0; padding-left: 1.2rem; display: grid; gap: .9rem; }
    .reference-item { color: var(--muted); line-height: 1.65; scroll-margin-top: 6rem; }
    .reference-authors, .reference-year, .reference-title { color: var(--ink); } .reference-title { font-style: italic; }
    .missing-asset { padding: 1.1rem; border-radius: .8rem; border: 1px dashed var(--border); color: var(--muted); font-family: var(--sans); }
    @media (max-width: 1024px) { .page-shell { grid-template-columns: 1fr; } .mobile-toggle { display: inline-flex; } .sidebar { position: fixed; inset: 0 auto 0 0; width: min(22rem,86vw); transform: translateX(-103%); transition: transform .25s ease; z-index: 25; } body.nav-open .sidebar { transform: translateX(0); } body.nav-open::after { content: ""; position: fixed; inset: 0; background: rgba(10,10,16,.32); z-index: 20; } .main { padding-top: 4.7rem; } }
    @media (max-width: 760px) { .hero { padding: 1.3rem 1rem; border-radius: 1.3rem; } .paper-body { padding: 0; } .paper-heading { flex-direction: column; gap: .25rem; } .heading-number, .heading-number-plain { min-width: auto; } p { font-size: 1rem; } table { min-width: 30rem; } }
  </style>
</head>
<body>
  <button class="mobile-toggle" type="button" aria-expanded="false" aria-controls="sidebar">Contents</button>
  <div class="page-shell">
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-top">
        <p class="eyebrow">Web Edition</p>
        <h2 class="sidebar-title">__SIDEBAR_TITLE__</h2>
        <div class="sidebar-meta">__SIDEBAR_META__</div>
        <div class="quick-links"><a href="downloads/phonosemantic_zenodo_final.pdf">PDF</a><a href="downloads/phonosemantic_zenodo.tex">LaTeX</a><a href="__DOI_URL__" target="_blank" rel="noreferrer">DOI</a></div>
      </div>
      <section class="theme-panel" aria-label="Theme settings">
        <strong>Theme</strong>
        <div class="theme-buttons"><button type="button" class="theme-button" data-theme-choice="auto">Auto</button><button type="button" class="theme-button" data-theme-choice="light">Light</button><button type="button" class="theme-button" data-theme-choice="dark">Dark</button></div>
        <p class="theme-help">Auto follows the visitor's browser or OS color preference.</p>
      </section>
      <nav class="nav-tree" aria-label="Paper navigation">__NAV__</nav>
    </aside>
    <main class="main">
      <article class="article">
        <header class="hero">
          <p class="eyebrow">Phonosemantics Research Paper</p>
          <h1>__TITLE__</h1>
          <div class="hero-meta"><span class="hero-pill">__AUTHOR__</span><span class="hero-pill">__DATE__</span><span class="hero-pill">Approx. __READ_TIME__ min read</span><span class="hero-pill">DOI __DOI__</span></div>
          <div class="hero-links"><a href="downloads/phonosemantic_zenodo_final.pdf">Open PDF</a><a href="downloads/phonosemantic_zenodo.tex">Source TeX</a><a href="__DOI_URL__" target="_blank" rel="noreferrer">Zenodo DOI</a></div>
        </header>
        <section class="paper-body">__CONTENT__</section>
      </article>
    </main>
  </div>
  <script>
    const THEME_KEY = "phonosemantics-theme";
    const mobileToggle = document.querySelector(".mobile-toggle");
    const themeButtons = Array.from(document.querySelectorAll("[data-theme-choice]"));
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");
    function applyTheme(choice) { const resolved = choice === "auto" ? (prefersDark.matches ? "dark" : "light") : choice; document.documentElement.dataset.theme = resolved; document.documentElement.dataset.themeChoice = choice; document.documentElement.style.colorScheme = resolved; themeButtons.forEach((button) => button.classList.toggle("is-active", button.dataset.themeChoice === choice)); }
    function loadTheme() { const stored = localStorage.getItem(THEME_KEY) || "auto"; applyTheme(stored); }
    themeButtons.forEach((button) => { button.addEventListener("click", () => { const choice = button.dataset.themeChoice; localStorage.setItem(THEME_KEY, choice); applyTheme(choice); }); });
    prefersDark.addEventListener("change", () => { if ((localStorage.getItem(THEME_KEY) || "auto") === "auto") { applyTheme("auto"); } });
    mobileToggle?.addEventListener("click", () => { const next = !document.body.classList.contains("nav-open"); document.body.classList.toggle("nav-open", next); mobileToggle.setAttribute("aria-expanded", String(next)); });
    document.addEventListener("click", (event) => { if (window.innerWidth > 1024 || !document.body.classList.contains("nav-open")) return; const sidebar = document.getElementById("sidebar"); if (!sidebar.contains(event.target) && event.target !== mobileToggle) { document.body.classList.remove("nav-open"); mobileToggle.setAttribute("aria-expanded", "false"); } });
    const navLinks = Array.from(document.querySelectorAll(".nav-link"));
    const headingMap = navLinks.map((link) => { const target = document.getElementById(link.dataset.target); return target ? { link, target } : null; }).filter(Boolean);
    const observer = new IntersectionObserver((entries) => { const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0]; if (!visible) return; headingMap.forEach(({ link, target }) => { link.classList.toggle("is-active", target === visible.target); }); }, { rootMargin: "-15% 0px -65% 0px", threshold: [0.15, 0.4, 0.7] });
    headingMap.forEach(({ target }) => observer.observe(target));
    navLinks.forEach((link) => { link.addEventListener("click", () => { if (window.innerWidth <= 1024) { document.body.classList.remove("nav-open"); mobileToggle?.setAttribute("aria-expanded", "false"); } }); });
    loadTheme();
  </script>
  <script>window.MathJax = { tex: { inlineMath: [["\\\\(","\\\\)"]], displayMath: [["\\\\[","\\\\]"]], packages: { "[+]": ["ams"] } }, options: { skipHtmlTags: ["script","noscript","style","textarea","pre"] } };</script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</body>
</html>
"""

    return (
        template
        .replace("__PAGE_TITLE__", escape(title_plain))
        .replace("__SIDEBAR_TITLE__", metadata["title"])
        .replace("__SIDEBAR_META__", metadata["author"] + "<br>" + metadata["date"])
        .replace("__DOI_URL__", escape(metadata["doi_url"]))
        .replace("__NAV__", nav_html)
        .replace("__TITLE__", metadata["title"])
        .replace("__AUTHOR__", metadata["author"])
        .replace("__DATE__", metadata["date"])
        .replace("__READ_TIME__", str(read_minutes))
        .replace("__DOI__", escape(metadata["doi"]))
        .replace("__CONTENT__", article_html)
    )


if __name__ == "__main__":
    build_site()
    print(f"Built standalone paper site at {OUTPUT_PATH}")
