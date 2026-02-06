from pathlib import Path
import shutil


def update_readme_relative_links():
    index = Path("docs/index.md")
    with index.open() as f:
        md = f.read()
        updated_md = md.replace("./docs/", "./")
        updated_md = updated_md.replace("./docs/", "./")
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
    rename_coverage_index()
    update_readme_relative_links()
