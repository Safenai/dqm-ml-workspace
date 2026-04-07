from nox import Session, options, param, parametrize
from nox_uv import session

options.error_on_external_run = True
options.default_venv_backend = "uv"
options.sessions = ["lint", "spell", "test", "type_check"]


@session(
    python=["3.12", "3.13"],
    uv_groups=["test"],
)
def compatibility(s: Session) -> None:
    s.run(
        "pytest",
        "tests/unit/core",
        *s.posargs,
    )
    s.run(
        "pytest",
        # Only quick tests for compatibility
        "tests/cli/test_job_cli.py",
        *s.posargs,
    )


@session(
    python=["3.12"],
    uv_groups=["test"],
)
def test(s: Session) -> None:
    s.run(
        "pytest",
        "--cov=packages/dqm-ml-job/src",
        "--cov=packages/dqm-ml-core/src",
        "--cov=packages/dqm-ml-pytorch/src",
        "--cov=packages/dqm-ml-images/src",
        "--cov=packages/dqm-ml-v2/src",
        "--cov-report=html:docs/reports/htmlcov",
        "--cov-report=term",
        "--cov-fail-under=1",
        "--html=docs/reports/pytest/pytest_report.html",
        "tests/unit",
        "tests/integration",
        "tests/cli",
        *s.posargs,
    )


@session(
    python=["3.12"],
    uv_groups=["test"],
)
def test_dev(s: Session) -> None:
    s.run(
        "pytest",
        "tests",
        *s.posargs,
    )


# For some sessions, set venv_backend="none" to simply execute scripts within the existing outer
# uv-generated virtual environment, rather than have nox create a new one for each session. This
# makes commonly repeated sessions execute faster.


@session(venv_backend="none")
@parametrize(
    "command",
    [
        param(
            [
                "ruff",
                "check",
                "packages",
                "--select",
                "I",
                # Also remove unused imports.
                "--select",
                "F401",
                "--extend-fixable",
                "F401",
                "--fix",
            ],
            id="sort_imports",
        ),
        param(["ruff", "format", "packages"], id="format"),
    ],
)
def fmt(s: Session, command: list[str]) -> None:
    s.run(*command)


@session(venv_backend="none")
@parametrize(
    "command",
    [
        param(["ruff", "check", "packages"], id="lint_check"),
        param(["ruff", "format", "--check", "packages"], id="format_check"),
        param(["ruff", "check", "tests"], id="lint_check"),
        param(["ruff", "format", "--check", "tests"], id="format_check"),
    ],
)
def lint(s: Session, command: list[str]) -> None:
    s.run(*command)


@session(venv_backend="none")
def lint_fix(s: Session) -> None:
    s.run(
        "ruff",
        "check",
        "packages/dqm-ml-job",
        "--extend-fixable",
        "F401",
        "--fix",
    )
    s.run(
        "ruff",
        "check",
        "packages/dqm-ml-core",
        "--extend-fixable",
        "F401",
        "--fix",
    )
    s.run(
        "ruff",
        "check",
        "packages/dqm-ml-images",
        "--extend-fixable",
        "F401",
        "--fix",
    )
    s.run(
        "ruff",
        "check",
        "packages/dqm-ml-pytorch",
        "--extend-fixable",
        "F401",
        "--fix",
    )
    s.run(
        "ruff",
        "check",
        "packages/dqm-ml-v2",
        "--extend-fixable",
        "F401",
        "--fix",
    )
    s.run(
        "ruff",
        "check",
        "tests",
        "--extend-fixable",
        "F401",
        "--fix",
    )


@session(venv_backend="none")
def type_check(s: Session) -> None:
    s.run("mypy", "packages/dqm-ml-job", "noxfile.py")
    s.run("mypy", "packages/dqm-ml-core", "noxfile.py")
    s.run("mypy", "packages/dqm-ml-images", "noxfile.py")
    s.run("mypy", "packages/dqm-ml-pytorch", "noxfile.py")
    s.run("mypy", "packages/dqm-ml-v2", "noxfile.py")


# Environment variable needed for mkdocstrings-python to locate source files.
doc_env = {"PYTHONPATH": "packages"}


@session(
    python=["3.12"],
    uv_groups=["docs"],
)
def docs(s: Session) -> None:
    s.run(
        "mkdocs",
        "build",
        "--strict",
        env=doc_env,
    )


@session(
    python=["3.12"],
    uv_groups=["docs"],
)
def docs_serve(s: Session) -> None:
    s.run(
        "mkdocs",
        "serve",
        env=doc_env,
    )


@session(
    python=["3.12"],
    uv_groups=["docs"],
)
def docs_offline(s: Session) -> None:
    s.run(
        "mkdocs",
        "build",
        "--no-strict",
        env=doc_env | {"MKDOCS_MATERIAL_OFFLINE": str(True)},
    )


@session(
    python=["3.12"],
    uv_groups=["docs"],
)
def docs_github_pages(s: Session) -> None:
    s.run("mkdocs", "gh-deploy", "--force", env=doc_env)


# Install only main dependencies for the license report.
@session(uv_groups=["licenses"], uv_no_install_project=True)
def licenses(s: Session) -> None:
    s.run("pip-licenses", *s.posargs)


@session(venv_backend="none")
def spell(s: Session) -> None:
    s.run("cspell", "lint", ".", *s.posargs)
