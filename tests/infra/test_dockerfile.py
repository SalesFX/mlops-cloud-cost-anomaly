"""Static validation tests for Dockerfile and .dockerignore.

These tests inspect the file contents — they do not require Docker to be installed.
They enforce the architectural decision that model artefacts are volume-mounted
at runtime and never baked into the image.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
DOCKERFILE_PATH = ROOT / "Dockerfile"
DOCKERIGNORE_PATH = ROOT / ".dockerignore"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE_PATH.read_text()


@pytest.fixture(scope="module")
def dockerignore() -> str:
    return DOCKERIGNORE_PATH.read_text()


def _instruction_lines(content: str, instruction: str) -> list[str]:
    """Return all non-comment lines that start with the given Docker instruction."""
    return [
        line.strip()
        for line in content.split("\n")
        if line.strip().upper().startswith(instruction.upper())
        and not line.strip().startswith("#")
    ]


def _ignore_patterns(content: str) -> set[str]:
    """Return the set of non-comment, non-empty patterns from a .dockerignore."""
    return {
        line.strip()
        for line in content.split("\n")
        if line.strip() and not line.strip().startswith("#")
    }


# ---------------------------------------------------------------------------
# Dockerfile — existence and structure
# ---------------------------------------------------------------------------


class TestDockerfileExists:
    def test_dockerfile_file_exists(self):
        assert DOCKERFILE_PATH.exists(), "Dockerfile not found at project root"


class TestDockerfileContent:
    def test_contains_uvicorn(self, dockerfile):
        assert "uvicorn" in dockerfile

    def test_exposes_port_8000(self, dockerfile):
        assert "EXPOSE 8000" in dockerfile

    def test_defines_model_path_env(self, dockerfile):
        assert "MODEL_PATH" in dockerfile

    def test_defines_model_metadata_path_env(self, dockerfile):
        assert "MODEL_METADATA_PATH" in dockerfile

    def test_defines_feature_schema_path_env(self, dockerfile):
        assert "FEATURE_SCHEMA_PATH" in dockerfile

    def test_defines_pythonunbuffered(self, dockerfile):
        assert "PYTHONUNBUFFERED" in dockerfile

    def test_defines_pythondontwritebytecode(self, dockerfile):
        assert "PYTHONDONTWRITEBYTECODE" in dockerfile

    def test_no_reload_flag(self, dockerfile):
        assert "--reload" not in dockerfile

    def test_copies_src(self, dockerfile):
        copy_lines = _instruction_lines(dockerfile, "COPY")
        assert any("src" in line for line in copy_lines)

    def test_copies_requirements(self, dockerfile):
        copy_lines = _instruction_lines(dockerfile, "COPY")
        assert any("requirements.txt" in line for line in copy_lines)

    def test_does_not_copy_models(self, dockerfile):
        """Model artefacts must NOT be baked into the image — they are volume-mounted."""
        copy_lines = _instruction_lines(dockerfile, "COPY")
        assert not any("models" in line for line in copy_lines), (
            "models/ must not be COPYed into the image. Mount as volume at runtime."
        )

    def test_creates_models_mount_point(self, dockerfile):
        """An empty /app/models directory should be pre-created as a clean mount point."""
        assert "models" in dockerfile and "mkdir" in dockerfile

    def test_uses_non_root_user(self, dockerfile):
        user_lines = _instruction_lines(dockerfile, "USER")
        assert any("appuser" in line for line in user_lines), (
            "Container should run as non-root user 'appuser'"
        )

    def test_last_user_is_not_root(self, dockerfile):
        user_lines = _instruction_lines(dockerfile, "USER")
        if user_lines:
            assert "root" not in user_lines[-1].lower()

    def test_uses_python_slim_image(self, dockerfile):
        from_lines = _instruction_lines(dockerfile, "FROM")
        assert any("python" in line and "slim" in line for line in from_lines)

    def test_uses_python_312(self, dockerfile):
        from_lines = _instruction_lines(dockerfile, "FROM")
        # Accept 3.12 or 3.12.x
        assert any("3.12" in line for line in from_lines)


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------


class TestDockerignoreExists:
    def test_dockerignore_file_exists(self):
        assert DOCKERIGNORE_PATH.exists(), ".dockerignore not found at project root"


class TestDockerignoreContent:
    def test_ignores_models(self, dockerignore):
        patterns = _ignore_patterns(dockerignore)
        assert "models/" in patterns or "models" in patterns, (
            "models/ must be in .dockerignore — artefacts are volume-mounted"
        )

    def test_ignores_git(self, dockerignore):
        patterns = _ignore_patterns(dockerignore)
        assert ".git" in patterns

    def test_ignores_venv(self, dockerignore):
        patterns = _ignore_patterns(dockerignore)
        assert ".venv" in patterns or "venv" in patterns

    def test_ignores_pycache(self, dockerignore):
        assert "__pycache__" in dockerignore

    def test_ignores_data(self, dockerignore):
        patterns = _ignore_patterns(dockerignore)
        assert "data/" in patterns

    def test_does_not_ignore_src(self, dockerignore):
        patterns = _ignore_patterns(dockerignore)
        assert "src" not in patterns and "src/" not in patterns, (
            "src/ must NOT be in .dockerignore"
        )

    def test_does_not_ignore_requirements_txt(self, dockerignore):
        patterns = _ignore_patterns(dockerignore)
        assert "requirements.txt" not in patterns, (
            "requirements.txt must NOT be in .dockerignore"
        )
