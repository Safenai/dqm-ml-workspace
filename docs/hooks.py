from pathlib import Path
import shutil


def update_readme_relative_links():
    index = Path("docs/index.md")
    with index.open() as f:
        md = f.read()
        # Replace various forms of docs/ paths
        updated_md = md.replace("./docs/", "./")
        updated_md = updated_md.replace("(docs/", "(")
        updated_md = updated_md.replace(
            "(packages/",
            "(https://github.com/Safenai/dqm-ml-workspace/tree/main/packages/",
        )
    with index.open("w") as f:
        f.write(updated_md)


def copy_example():
    Path("docs/examples").mkdir(exist_ok=True)
    shutil.copy(
        "examples/multiple_metrics_tests_v2.ipynb",
        "docs/examples/multiple_metrics_tests_v2.ipynb",
    )


def copy_readme():
    shutil.copy("README.md", "docs/index.md")
    shutil.copy("AGENTS.md", "docs/AGENTS.md")


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
        "dqm-ml-v2",
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


def pre_build(*args, **kwargs):
    copy_example()
    copy_readme()
    copy_package_readmes()
    rename_coverage_index()
    update_readme_relative_links()
