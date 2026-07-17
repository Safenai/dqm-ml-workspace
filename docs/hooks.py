import re
import shutil
from pathlib import Path
from urllib.parse import urljoin

DOCS_INDEX = "docs/index.md"
REPO_URL = "https://github.com/Safenai/dqm-ml-workspace"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _copy_if_changed(src: Path, dst: Path):
    """Copy file only if destination doesn't exist or content differs.

    Args:
        src: Source file path.
        dst: Destination file path.
    """
    if dst.exists() and dst.read_bytes() == src.read_bytes():
        return
    shutil.copy(src, dst)


def _write_if_changed(dst: Path, content: str):
    """Write text content to file only if it differs from existing content.

    Args:
        dst: Destination file path.
        content: Text content to write.
    """
    if dst.exists() and dst.read_bytes() == content.encode():
        return
    dst.write_text(content)


# ---------------------------------------------------------------------------
# docs/index.md pipeline  (single pass — no ordering dependencies)
# ---------------------------------------------------------------------------


def _build_index():
    """Copy README.md to docs/index.md, rewrite links for mkdocs site."""
    content = Path("README.md").read_text()

    # Rewrite relative paths that point into docs/ for GitHub/GitLab
    content = content.replace("./docs/", "./")
    content = content.replace("(docs/", "(")
    content = content.replace('src="docs/static/', 'src="./static/')
    content = content.replace(
        "(packages/",
        f"({REPO_URL}/tree/main/packages/",
    )
    content = content.replace(
        "(examples/",
        f"({REPO_URL}/tree/main/examples/",
    )

    # Inject repository link before "Available on PyPI" if not already present
    repo_link = "- **[Repository](https://github.com/Safenai/dqm-ml-workspace)**"
    if "## Available on PyPI" in content and repo_link not in content:
        content = content.replace(
            "## Available on PyPI",
            f"{repo_link}\n\n## Available on PyPI",
        )

    _write_if_changed(Path(DOCS_INDEX), content)


# ---------------------------------------------------------------------------
# docs/examples/ pipeline  (single pass — copy + link fix in one go)
# ---------------------------------------------------------------------------


def _apply_example_link_fixes(content: str, depth: int) -> str:
    """Rewrite relative links in example markdown based on destination depth.

    Source files use ``../docs/...`` paths that work on GitHub. After copying
    into ``docs/examples/``, the ``docs/`` prefix must be dropped.

    Args:
        content: Markdown content to transform.
        depth: Directory depth in docs/examples/ (0 = root, 1 = subdirectory).

    Returns:
        Markdown content with relative links rewritten.
    """
    if depth == 0:
        return content.replace("../docs/metrics/", "../metrics/")
    elif depth == 1:
        return content.replace("../../docs/metrics/", "../../metrics/")
    return content


def _copy_md_with_fixes(src: Path, dst: Path, depth: int):
    """Copy a markdown file, applying link fixes in memory before writing.

    Args:
        src: Source markdown file path.
        dst: Destination file path.
        depth: Directory depth for link fix calculations.
    """
    content = _apply_example_link_fixes(src.read_text(), depth)
    _write_if_changed(dst, content)


def _sync_dir(src: Path, dst: Path, *, fix_md_links: bool = False):
    """Sync a single directory by copying files to a destination.

    Args:
        src: Source directory path.
        dst: Destination directory path.
        fix_md_links: If True, apply link fixes when copying markdown files.
    """
    if not src.exists():
        return
    dst.mkdir(exist_ok=True)
    for src_file in sorted(src.iterdir()):
        if src_file.is_dir():
            continue
        dst_file = dst / src_file.name
        if src_file.suffix == ".md" and fix_md_links:
            _copy_md_with_fixes(src_file, dst_file, depth=0)
        else:
            _copy_if_changed(src_file, dst_file)


def _build_examples():
    """Copy examples/ to docs/examples/, rewriting .md links in one pass."""
    src_root = Path("examples")
    dst_root = Path("docs/examples")
    dst_root.mkdir(exist_ok=True)

    _sync_dir(src_root / "scenario", dst_root / "scenario", fix_md_links=True)
    _sync_dir(src_root / "notebooks", dst_root / "notebooks")
    _sync_dir(src_root / "config", dst_root / "config")
    _sync_dir(src_root / "config" / "scenario", dst_root / "config" / "scenario")
    _sync_dir(src_root / "script", dst_root / "script")

    overview = src_root / "overview.md"
    if overview.exists():
        _copy_md_with_fixes(overview, dst_root / "overview.md", depth=0)


# ---------------------------------------------------------------------------
# Simple file copies  (source root → docs/)
# ---------------------------------------------------------------------------


def copy_changelog():
    """Copy CHANGELOG.md from repo root to docs/."""
    src = Path("CHANGELOG.md")
    if src.exists():
        _copy_if_changed(src, Path("docs/CHANGELOG.md"))


def copy_release_notes():
    """Copy RELEASE.md from repo root to docs/."""
    src = Path("RELEASE.md")
    if src.exists():
        _copy_if_changed(src, Path("docs/RELEASE.md"))


def copy_package_readmes():
    """Copy package READMEs to docs/packages/."""
    packages_dir = Path("packages")
    docs_packages_dir = Path("docs/packages")
    docs_packages_dir.mkdir(exist_ok=True)

    for pkg in [
        "dqm-ml-core",
        "dqm-ml-job",
        "dqm-ml-images",
        "dqm-ml-pytorch",
        "dqm-ml",
    ]:
        src = packages_dir / pkg / "README.md"
        if src.exists():
            _copy_if_changed(src, docs_packages_dir / f"{pkg}.md")


def _copy_coverage_report():
    """Copy coverage index.html to coverage_report.html for direct linking."""
    coverage_index = Path("docs/reports/htmlcov/index.html")
    if coverage_index.exists():
        _copy_if_changed(coverage_index, Path("docs/reports/htmlcov/coverage_report.html"))


# ---------------------------------------------------------------------------
# mkdocs page_markdown hook  (per-page, during render)
# ---------------------------------------------------------------------------


def _transform_examples_to_github(markdown: str, src_path: str) -> str:
    """Transform relative links to examples/ into GitHub URLs.

    Source docs/*.md files use relative paths like ``../examples/...`` that
    work locally and on GitHub/GitLab. On the mkdocs website these example
    files aren't served directly, so rewrite the links to permanent GitHub
    URLs.

    Args:
        markdown: Raw markdown content.
        src_path: Source path of the page relative to docs/.

    Returns:
        Markdown content with example links rewritten to GitHub URLs.
    """
    github_raw = f"{REPO_URL}/tree/main"

    if src_path.startswith("examples/"):
        page_dir = src_path.rsplit("/", 1)[0] + "/"

        def _repl(m):
            text, url = m.group(1), m.group(2)
            if url.startswith(("https://", "#", "/")):
                return m.group(0)
            if url.endswith(".md"):
                return m.group(0)
            if url.endswith((".py", ".yaml")):
                resolved = urljoin(page_dir, url)
                return f"[{text}]({github_raw}/{resolved})"
            return m.group(0)

        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _repl, markdown)

    # src_path is relative to docs/ directory.
    # E.g., "cli.md" → depth 0 → need "../" to reach repo root
    #        "configuration/features.md" → depth 1 → need "../../"
    depth = src_path.count("/")
    prefix = "../" * (depth + 1)
    return markdown.replace(f"({prefix}examples/", f"({github_raw}/examples/")


def _transform_package_links(markdown: str, src_path: str) -> str:
    """Transform relative package README links into docs/ page links.

    Source docs/*.md files use ``../packages/xxx/README.md`` paths that work
    on GitHub. On the mkdocs website these READMEs have been copied to
    ``docs/packages/xxx.md``, so rewrite the links accordingly.

    Args:
        markdown: Raw markdown content.
        src_path: Source path of the page relative to docs/.

    Returns:
        Markdown content with package links rewritten.
    """

    if src_path.startswith("examples/"):
        return markdown

    return re.sub(
        r"\(\.\./packages/([^/]+)/README\.md(#[^)]*)?\)",
        r"(packages/\1.md\2)",
        markdown,
    )


def page_markdown(markdown, page, config, files):  # NOSONAR
    """Transform relative links after page markdown is loaded.

    Args:
        markdown: Raw markdown content of the page.
        page: MkDocs page object.
        config: MkDocs configuration.
        files: Collection of all files in the docs directory.

    Returns:
        Transformed markdown content.
    """
    markdown = _transform_examples_to_github(markdown, page.file.src_path)
    markdown = _transform_package_links(markdown, page.file.src_path)
    return markdown


# ---------------------------------------------------------------------------
# mkdocs pre_build hook  (runs once before every build)
# ---------------------------------------------------------------------------


def pre_build(*args, **kwargs):
    """Pre-build hook: copy and transform files for mkdocs.

    All functions are self-contained — no implicit ordering dependencies.
    Each uses content-aware writes to avoid triggering unnecessary rebuilds.
    """
    copy_changelog()
    copy_release_notes()
    copy_package_readmes()
    _copy_coverage_report()
    _build_index()
    _build_examples()
