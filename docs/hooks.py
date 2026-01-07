from pathlib import Path
import shutil


def update_readme_relative_links():
    index = Path("docs/index.md")
    with index.open() as f:
        md = f.read()
        updated_md=md.replace('./docs/', './')
        updated_md=updated_md.replace('./docs/', './')
        updated_md=updated_md.replace(
            '(packages/',
            '(https://github.com/Safenai/dqm-ml-workspace/tree/main/packages/'
        )
    with index.open("w") as f:
        f.write(updated_md)

def copy_example():
    Path.mkdir("docs/examples", exist_ok=True)
    shutil.copy(
        "packages/dqm-ml/examples/multiple_metrics_tests_v2.ipynb", 
        "docs/examples/multiple_metrics_tests_v2.ipynb"
    )

def copy_readme():
    shutil.copy("README.md", "docs/index.md")

def pre_build(*args, **kwargs):
    copy_example()
    copy_readme()
    update_readme_relative_links()