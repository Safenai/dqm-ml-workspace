from nox import Session, options, param, parametrize
from nox_uv import session

options.error_on_external_run = True
options.default_venv_backend = "uv"
options.sessions = ["lint", "test", "type_check"]


@session(
    python=["3.9", "3.10", "3.11", "3.12", "3.13"],
    uv_groups=["test"],
)
def compatibility(s: Session) -> None:
    s.run(
        "pytest",
        "packages/dqm-ml-core/tests",
        *s.posargs,
    )
    s.run(
        "pytest",
        # Only quick tests for compatibility
        "packages/dqm-ml-pipeline/tests/test_cli.py",
        *s.posargs,
    )
    # TODO reactivate when tests are availables
    # s.run(
    #    "pytest",
    #    "packages/dqm-ml/tests",
    #    *s.posargs,
    # )


@session(
    python=["3.12"],
    uv_groups=["test"],
)
def test(s: Session) -> None:
    s.run(
        "pytest",
        "--cov=packages/dqm-ml-core/src",
        "--cov-fail-under=1",
        "packages/dqm-ml-core/tests",
        *s.posargs,
    )
    s.run(
        "pytest",
        "--cov-append",
        "--cov=packages/dqm-ml-pipeline/src",
        "--cov=packages/dqm-ml-core/src",
        "--cov-report=html",
        "--cov-report=term",  
        "--cov-fail-under=1",
        "packages/dqm-ml-pipeline/tests",
        *s.posargs,
    )

    # TODO re-enable when tests are available
    # s.run(
    #    "pytest",
    #    "--cov-append",
    #    "--cov=packages/dqm-ml-pipeline/src",
    #    "--cov=packages/dqm-ml-core/src",
    #    "--cov=packages/dqm-ml/src",
    #    "--cov-report=html",
    #    "--cov-report=term",
    #    "--cov-fail-under=1",
    #    "packages/dqm-ml-v2/tests",
    #    *s.posargs,
    # )


@session(
    python=["3.12"],
    uv_groups=["test"],
)
def test_dev(s: Session) -> None:
    s.run(
        "pytest",
        "packages/dqm-ml-pipeline/tests",
        *s.posargs,
    )

# TODO add a way to iterate on package and update level to 100%

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
                ".",
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
    ],
)
def lint(s: Session, command: list[str]) -> None:
    s.run(*command)


@session(venv_backend="none")
def lint_fix(s: Session) -> None:
    s.run("ruff", "check", "packages/dqm-ml-pipeline", "--extend-fixable", "F401", "--fix")
    s.run("ruff", "check", "packages/dqm-ml-core", "--extend-fixable", "F401", "--fix")
    s.run("ruff", "check", "packages/dqm-ml-images", "--extend-fixable", "F401", "--fix")
    s.run("ruff", "check", "packages/dqm-ml-pytorch", "--extend-fixable", "F401", "--fix")


@session(venv_backend="none")
def type_check(s: Session) -> None:
    s.run("mypy", "packages/dqm-ml-pipeline", "noxfile.py")
    s.run("mypy", "packages/dqm-ml-core", "noxfile.py")
    s.run("mypy", "packages/dqm-ml-images", "noxfile.py")
    s.run("mypy", "packages/dqm-ml-pytorch", "noxfile.py")


# Install only main dependencies for the license report.
@session(uv_groups=["licenses"], uv_no_install_project=True)
def licenses(s: Session) -> None:
    s.run("pip-licenses", *s.posargs)
