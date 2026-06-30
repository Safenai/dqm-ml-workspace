from pathlib import Path
import shutil

DOCS_INDEX = "docs/index.md"


def update_readme_relative_links():
    index = Path(DOCS_INDEX)
    with index.open() as f:
        md = f.read()
        # Replace various forms of docs/ paths
        updated_md = md.replace("./docs/", "./")
        updated_md = updated_md.replace("(docs/", "(")
        updated_md = updated_md.replace('src="docs/static/', 'src="./static/')
        updated_md = updated_md.replace(
            "(packages/",
            "(https://github.com/Safenai/dqm-ml-workspace/tree/main/packages/",
        )
    with index.open("w") as f:
        f.write(updated_md)


def copy_examples():
    src_root = Path("examples")
    dst_root = Path("docs/examples")
    dst_root.mkdir(exist_ok=True)

    # Copy overview.md from root
    overview = src_root / "overview.md"
    if overview.exists():
        shutil.copy(overview, dst_root / "overview.md")

    # Copy scenario markdowns
    scenario_src = src_root / "scenario"
    if scenario_src.exists():
        scenario_dst = dst_root / "scenario"
        scenario_dst.mkdir(exist_ok=True)
        for md_file in scenario_src.glob("*.md"):
            shutil.copy(md_file, scenario_dst / md_file.name)

    # Copy notebooks
    notebooks_src = src_root / "notebooks"
    if notebooks_src.exists():
        notebooks_dst = dst_root / "notebooks"
        notebooks_dst.mkdir(exist_ok=True)
        for nb_file in notebooks_src.glob("*.ipynb"):
            shutil.copy(nb_file, notebooks_dst / nb_file.name)

    # Copy configs
    config_dst = dst_root / "config"
    config_dst.mkdir(exist_ok=True)
    for yaml_file in (src_root / "config").glob("*.yaml"):
        shutil.copy(yaml_file, config_dst / yaml_file.name)

    # Copy configs
    config_dst = dst_root / "config/scenario"
    config_dst.mkdir(exist_ok=True)
    for yaml_file in (src_root / "config/scenario").glob("*.yaml"):
        shutil.copy(yaml_file, config_dst / yaml_file.name)

    # Copy scripts
    script_dst = dst_root / "script"
    script_dst.mkdir(exist_ok=True)
    shutil.copy(
        src_root / "script" / "completeness.py",
        script_dst / "completeness.py",
    )


def copy_readme():
    shutil.copy("README.md", DOCS_INDEX)


def copy_package_readmes():
    """Copy package READMEs to docs/packages/ for documentation."""
    packages_dir = Path("packages")
    docs_packages_dir = Path("docs/packages")
    docs_packages_dir.mkdir(exist_ok=True)

    packages_to_copy = [
        "dqm-ml-core",
        "dqm-ml-job",
        "dqm-ml-images",
        "dqm-ml-pytorch",
        "dqm-ml",
    ]

    for pkg in packages_to_copy:
        src = packages_dir / pkg / "README.md"
        if src.exists():
            dst = docs_packages_dir / f"{pkg}.md"
            shutil.copy(src, dst)


def rename_coverage_index():
    coverage_index = Path("docs/reports/htmlcov/index.html")
    if coverage_index.exists():
        shutil.copy(
            "docs/reports/htmlcov/index.html",
            "docs/reports/htmlcov/coverage_report.html",
        )


def add_repository_link():
    """Add repository link before Available on PyPI section."""
    index = Path(DOCS_INDEX)
    with index.open() as f:
        md = f.read()

    repository_link = "- **[Repository](https://github.com/Safenai/dqm-ml-workspace)**"

    if "## Available on PyPI" in md and repository_link not in md:
        updated_md = md.replace("## Available on PyPI", f"{repository_link}\n\n## Available on PyPI")
        with index.open("w") as f:
            f.write(updated_md)


def fix_example_links():
    """Rewrite relative links in copied examples so they work under docs/examples/.

    Source examples/ files use ../docs/... or ../../docs/... links that work
    locally and on GitHub/GitLab. After being copied to docs/examples/, the
    links need adjusting based on directory depth:
      - docs/examples/ (depth 0):  ../docs/metrics/  -> ../metrics/
      - docs/examples/scenario/ (depth 1): ../../docs/metrics/ -> ../../metrics/
    """
    examples_dir = Path("docs/examples")
    for md_file in examples_dir.rglob("*.md"):
        rel = md_file.relative_to(examples_dir)
        depth = len(rel.parent.parts)

        with md_file.open() as f:
            md = f.read()

        if depth == 0:
            updated_md = md.replace("../docs/metrics/", "../metrics/")
        elif depth == 1:
            updated_md = md.replace("../../docs/metrics/", "../../metrics/")
        else:
            updated_md = md

        if updated_md != md:
            with md_file.open("w") as f:
                f.write(updated_md)


def page_markdown(markdown, page, config, files):
    """Rewrite ../examples/ links to examples/ for mkdocs.

    Source docs/configuration.md uses ../examples/... links that work
    locally and on GitHub/GitLab (pointing to examples/ at repo root).
    On mkdocs the examples are at docs/examples/, so rewrite the links
    to examples/... (without the ../ prefix) at build time.
    """
    if page.file.src_path == "configuration.md":
        markdown = markdown.replace("../examples/", "examples/")
    return markdown


def pre_build(*args, **kwargs):
    copy_examples()
    fix_example_links()
    copy_readme()
    add_repository_link()
    copy_package_readmes()
    rename_coverage_index()
    update_readme_relative_links()
