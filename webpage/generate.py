"""Generate the public Biogeme webpage from its source and Sphinx output."""

from __future__ import annotations

import re
import shutil
import tempfile
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import tomlkit as tk
from markdown_it import MarkdownIt

from biogeme.version import __version__

try:  # Support both ``python webpage/generate.py`` and package imports.
    from .faq import faq
    from .sections import about, archives, documentation, install, resources, special
except ImportError:  # pragma: no cover - exercised by the script entry point.
    from faq import faq
    from sections import about, archives, documentation, install, resources, special


ROOT = Path(__file__).resolve().parent
SPHINX_BUILD = (ROOT / "../docs/build/html").resolve()
TARGET_FILE = "index.html"
DATA_FILE = ROOT / "data.toml"
RELEASE_NOTES_FILE = ROOT.parent / "RELEASE_NOTES.md"
PORTFOLIO_MODAL = ROOT / "portfolio_modal.html"
PORTFOLIO_FILE = ROOT / "portfolio_grid.html.orig"
PORTFOLIO_GRID_ITEM = ROOT / "portfolio_grid_item.html"
HTML_FILE = ROOT / "index.html.orig"
FAQ_FILE = ROOT / "faq.html"
CARD_FILE = ROOT / "card.html"
SPECIAL_FILE = ROOT / "special.html"

_PLAIN_VERSION = re.compile(r"\A\d+\.\d+\.\d+\Z")
_PLACEHOLDER = re.compile(r"__[A-Z][A-Z0-9_]*__|__[A-Z][A-Z0-9_]*\Z")
_BLOCK_ELEMENT = re.compile(
    r"\A\s*<(?:address|article|aside|blockquote|details|div|dl|fieldset|figcaption|figure|footer|form|h[1-6]|header|hr|iframe|li|main|nav|ol|p|pre|section|table|ul)\b",
    re.IGNORECASE | re.DOTALL,
)
_LOCAL_SCHEMES = {"", "file"}
_ASSET_IGNORED_NAMES = {"README"}
_FORBIDDEN_PATH_TEXT = ("file://", "/Users/bierlair/", "/home/bierlair/", "CloudStorage/")
_RELEASE_HEADING = re.compile(
    r"^# Biogeme (?P<version>\d+\.\d+\.\d+)\s*$", re.MULTILINE
)


def validate_release_version(version: str) -> str:
    """Return a publishable version or fail for a provisional version.

    The public webpage is only generated for a plain ``major.minor.patch``
    release.  Alpha, beta, release-candidate, development, post, and local
    versions must not accidentally be published as the official webpage.
    """

    normalized = str(version).strip()
    if not _PLAIN_VERSION.fullmatch(normalized):
        raise RuntimeError(
            "Webpage generation requires a plain release version such as "
            f"3.3.4; got {normalized!r}. Alpha, beta, release-candidate, "
            "development, post, and local versions are not publishable."
        )
    return normalized


BIOGEME_VERSION = validate_release_version(__version__)


def release_notes_title() -> str:
    """Return the title used for the current release notes."""

    return f"What's new in Biogeme {BIOGEME_VERSION}?"


def load_release_note_sections() -> list[tuple[str, str]]:
    """Load and render all release-note sections from the canonical Markdown.

    The first heading identifies the current release and is required to match
    the package version. Subsequent headings are historical FAQ entries.
    """

    if not RELEASE_NOTES_FILE.is_file():
        raise RuntimeError(f"Release notes file is missing: {RELEASE_NOTES_FILE}")
    source = RELEASE_NOTES_FILE.read_text(encoding="utf-8")
    expected_heading = f"# Biogeme {BIOGEME_VERSION}"
    matches = list(_RELEASE_HEADING.finditer(source))
    if not matches or matches[0].group(0).strip() != expected_heading:
        raise RuntimeError(
            f"Release notes must begin with {expected_heading!r}; "
            f"got {matches[0].group(0).strip()!r}"
            if matches
            else f"got no release heading in {RELEASE_NOTES_FILE}."
        )

    sections: list[tuple[str, str]] = []
    markdown = MarkdownIt("commonmark", {"html": True})
    seen_versions: set[str] = set()
    for index, match in enumerate(matches):
        version = match.group("version")
        if version in seen_versions:
            raise RuntimeError(f"Duplicate release-note section for {version}.")
        seen_versions.add(version)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        body = source[match.end() : end].strip()
        if not body:
            raise RuntimeError(f"Release notes section for {version} is empty.")
        rendered = markdown.render(body).strip()
        if not rendered:
            raise RuntimeError(
                f"Release notes section for {version} renders to empty HTML."
            )
        sections.append((version, rendered))
    return sections


def load_release_notes() -> str:
    """Return the rendered notes for the current release."""

    return load_release_note_sections()[0][1]


def with_current_release_notes(content: dict, notes: str) -> dict:
    """Return *content* with the current release notes inserted first."""

    title = release_notes_title()
    historical = {key: value for key, value in content.items() if key != title}
    return {title: notes, **historical}


def with_historical_release_notes(
    content: dict[str, str], sections: list[tuple[str, str]]
) -> dict[str, str]:
    """Insert archived release-note sections before ordinary FAQ entries."""

    historical = {
        f"What's new in Biogeme {version}?": notes
        for version, notes in sections[1:]
    }
    remaining = {key: value for key, value in content.items() if key not in historical}
    return {**historical, **remaining}


def replace(orig_text: str, dictionary: dict[str, str]) -> str:
    """Replace generator tokens in a template."""

    for key, value in dictionary.items():
        orig_text = orig_text.replace(key, value)
    return orig_text


def _render_card_paragraph(paragraph: str) -> str:
    """Render prose in a card without wrapping block HTML in ``<p>``."""

    if _BLOCK_ELEMENT.match(paragraph):
        return f"{paragraph.rstrip()}\n"
    return f'<p class="card-text">{paragraph}</p>\n'


def get_section(content: dict[str, tuple[str, ...] | str]) -> str:
    """Render the cards for one homepage section."""

    all_html = ""
    for card_title, card_paragraphs in content.items():
        with CARD_FILE.open(encoding="utf-8") as file:
            html = file.read()
        if isinstance(card_paragraphs, str):
            paragraphs = (card_paragraphs,)
        else:
            paragraphs = card_paragraphs
        text_cards = "".join(_render_card_paragraph(paragraph) for paragraph in paragraphs)
        replacements_section = {
            "__TITLE__": card_title,
            "__CONTENT__": text_cards,
            "__VERSION__": BIOGEME_VERSION,
        }
        all_html += replace(html, replacements_section) + "\n"
    return all_html


def get_faq(content: dict[str, str] | None = None) -> str:
    """Render all FAQ entries."""

    all_html = ""
    entries = faq if content is None else content
    for item_id, (question, answer) in enumerate(entries.items(), start=1):
        with FAQ_FILE.open(encoding="utf-8") as file:
            html = file.read()
        replacements = {
            "__ID__": str(item_id),
            "__QUESTION__": question,
            "__ANSWER__": answer,
        }
        all_html += replace(html, replacements) + "\n"
    return all_html


def get_special(content: dict[str, str]) -> str:
    """Render release and other special announcements."""

    all_html = ""
    with SPECIAL_FILE.open(encoding="utf-8") as file:
        template = file.read()

    for special_title, special_content in content.items():
        replacements = {
            "__TITLE__": special_title,
            "__CONTENT__": special_content,
            "__VERSION__": BIOGEME_VERSION,
        }
        all_html += replace(template, replacements)
    return all_html


def get_portfolio_grid(doc: dict) -> str:
    """Render the dataset portfolio grid."""

    all_html = ""
    for data, values in doc.items():
        with PORTFOLIO_GRID_ITEM.open(encoding="utf-8") as file:
            html = file.read()
        replacements = {
            "__ID__": data,
            "__TITLE__": values["title"],
            "__IMAGE__": values["picture"],
            "__SHORT__": values["short_description"],
            "__LONG__": values["long_description"],
            "__PDF__": values["pdf_file"],
            "__DATA__": values["data_file"],
        }
        all_html += replace(html, replacements) + "\n"
    return all_html


def get_portfolio_modals(doc: dict) -> str:
    """Render the dataset portfolio modals."""

    all_html = ""
    for data, values in doc.items():
        with PORTFOLIO_MODAL.open(encoding="utf-8") as file:
            html = file.read()
        replacements = {
            "__ID__": data,
            "__TITLE__": values["title"],
            "__IMAGE__": values["picture"],
            "__SHORT__": values["short_description"],
            "__LONG__": values["long_description"],
            "__PDF__": values["pdf_file"],
            "__DATA__": values["data_file"],
        }
        all_html += replace(html, replacements) + "\n"
    return all_html


def build_homepage() -> str:
    """Expand the homepage templates and source dictionaries."""

    release_sections = load_release_note_sections()
    release_notes = release_sections[0][1]
    homepage_about = with_current_release_notes(about, release_notes)
    homepage_faq = with_current_release_notes(
        with_historical_release_notes(faq, release_sections), release_notes
    )

    with DATA_FILE.open(encoding="utf-8") as file:
        doc = tk.parse(file.read())

    with HTML_FILE.open(encoding="utf-8") as file:
        html = file.read()

    with PORTFOLIO_FILE.open(encoding="utf-8") as file:
        portfolio_html = file.read()

    portfolio_html = replace(
        portfolio_html,
        {
            "__PORTTITLE__": "Data",
            "__PORTDESC__": (
                "We provide here some choice data sets that can be used "
                "for research and education."
            ),
            "__CONTENT__": get_portfolio_grid(doc),
        },
    )

    replacements = {
        "__PORTFOLIO_MODALS__": get_portfolio_modals(doc),
        "__PORTFOLIO__": portfolio_html,
        "__FAQ__": get_faq(homepage_faq),
        "__ABOUT__": get_section(homepage_about),
        "__INSTALL__": get_section(install),
        "__DOC__": get_section(documentation),
        "__RES__": get_section(resources),
        "__ARCHIVES__": get_section(archives),
        "__SPECIAL__": get_special(special),
        "__VERSION__": BIOGEME_VERSION,
        "__YEAR__": str(date.today().year),
    }
    return replace(html, replacements)


class _LinkCollector(HTMLParser):
    """Collect links and element IDs from one HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        for attribute in ("href", "src"):
            value = attributes.get(attribute)
            if value:
                self.links.append(value)
        element_id = attributes.get("id")
        if element_id:
            self.ids.add(element_id)
        name = attributes.get("name")
        if name:
            self.ids.add(name)


def _resolve_local_link(document: Path, link: str, website: Path) -> tuple[Path, str]:
    """Resolve a local link against its document and return its fragment."""

    parsed = urlsplit(link)
    without_fragment = unquote(parsed.path)
    fragment = parsed.fragment
    if without_fragment.startswith("/"):
        target = website / without_fragment.lstrip("/")
    else:
        target = document.parent / without_fragment
    target = target.resolve()
    if target.is_dir():
        target /= "index.html"
    return target, fragment


def validate_generated_website(website: Path, sphinx_build: Path) -> None:
    """Validate generated links, placeholders, and release-local leakage."""

    html_files = sorted(website.rglob("*.html"))
    if not html_files:
        raise RuntimeError(f"Generated website contains no HTML files: {website}")

    errors: list[str] = []
    target_ids: dict[Path, set[str]] = {}
    for document in html_files:
        text = document.read_text(encoding="utf-8")
        relative_document = document.relative_to(website)
        # Sphinx pages intentionally contain source-code examples with Python
        # placeholders and sample filesystem paths.  Validate those pages for
        # links, but apply release-content checks only to webpage-owned HTML.
        if relative_document.parts[0] != "sphinx":
            for match in _PLACEHOLDER.finditer(text):
                errors.append(f"{relative_document}: unresolved {match.group()}")
            for forbidden in _FORBIDDEN_PATH_TEXT:
                if forbidden in text:
                    errors.append(f"{relative_document}: contains {forbidden!r}")

        parser = _LinkCollector()
        parser.feed(text)
        for link in parser.links:
            parsed = urlsplit(link)
            if parsed.scheme not in _LOCAL_SCHEMES or parsed.netloc:
                continue
            if link.startswith("//") or link.startswith("#"):
                continue
            target, fragment = _resolve_local_link(document, link, website)
            if not target.is_file():
                errors.append(f"{relative_document}: missing {link}")
                continue
            if fragment:
                if target not in target_ids:
                    target_parser = _LinkCollector()
                    target_parser.feed(target.read_text(encoding="utf-8"))
                    target_ids[target] = target_parser.ids
                if fragment not in target_ids[target]:
                    errors.append(
                        f"{relative_document}: missing fragment {link}"
                    )

    if not (website / "sphinx" / "index.html").is_file():
        errors.append("missing copied Sphinx documentation: sphinx/index.html")
    if sphinx_build.resolve() == website.resolve():
        errors.append("Sphinx source and generated website paths must be different")

    if errors:
        formatted = "\n".join(f"- {error}" for error in errors[:40])
        more = "" if len(errors) <= 40 else f"\n- ... and {len(errors) - 40} more"
        raise RuntimeError(f"Generated webpage validation failed:\n{formatted}{more}")


def _ignore_nonproduction_assets(_directory: str, names: list[str]) -> set[str]:
    """Exclude source-only templates and notes from the public asset tree."""

    return {
        name
        for name in names
        if Path(name).suffix.lower() in {".html", ".orig"}
        or name in _ASSET_IGNORED_NAMES
    }


def generate() -> Path:
    """Generate and validate ``website/`` atomically."""

    if not SPHINX_BUILD.is_dir() or not (SPHINX_BUILD / "index.html").is_file():
        raise RuntimeError(
            f"Sphinx output is missing or incomplete: {SPHINX_BUILD}. "
            "Build the documentation before generating the webpage."
        )

    website = ROOT / "website"
    backup = ROOT / "website.old"
    staging = Path(tempfile.mkdtemp(prefix=".website-", dir=ROOT))
    try:
        (staging / "index.html").write_text(build_homepage(), encoding="utf-8")
        shutil.copytree(ROOT / "js", staging / "js")
        shutil.copytree(ROOT / "css", staging / "css")
        shutil.copytree(
            ROOT / "assets",
            staging / "assets",
            ignore=_ignore_nonproduction_assets,
        )
        shutil.copytree(SPHINX_BUILD, staging / "sphinx")
        validate_generated_website(staging, SPHINX_BUILD)

        shutil.rmtree(backup, ignore_errors=True)
        if website.exists():
            website.rename(backup)
        staging.rename(website)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return website


def main() -> None:
    """Generate the webpage when invoked as a script."""

    website = generate()
    print(f"Generated and validated {website}")


if __name__ == "__main__":
    main()
