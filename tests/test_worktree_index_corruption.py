"""E2E tests: git worktree index corruption caused by GIT_INDEX_FILE inheritance.

During git commit in a worktree, git sets GIT_INDEX_FILE to the worktree's
index path. claude -p inherits this env var, and Claude Code's plugin init
runs git operations that overwrite the worktree's index with plugin cache
entries. Result: "error: Error building trees" and a destroyed index.

Test A: Pre-commit hook calls claude -p directly.
Test B: Pre-commit hook calls plumb hook (the real code path).

All tests: real git, real claude -p, real worktree, real commit. No mocks.
Marked slow — requires claude CLI installed and authenticated.

See: https://github.com/ktinubu/plumb/issues/1
"""

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest
from git import Repo

from plumb.config import PlumbConfig, save_config, ensure_plumb_dir

needs_claude_cli = pytest.mark.skipif(
    shutil.which("claude") is None,
    reason="claude CLI not installed",
)


def _create_repo_with_worktree(tmp_path, num_files=20):
    """Create a main repo with files and a worktree.

    Returns (main_repo_path, worktree_path).
    """
    main_dir = tmp_path / "main-repo"
    main_dir.mkdir()
    repo = Repo.init(main_dir)

    for i in range(num_files):
        (main_dir / f"file_{i}.txt").write_text(f"content {i}\n")
    repo.index.add([f"file_{i}.txt" for i in range(num_files)])
    repo.index.commit("initial commit")

    wt_dir = tmp_path / "worktree"
    repo.git.worktree("add", str(wt_dir), "-b", "wt-branch", "HEAD")

    return main_dir, wt_dir


def _count_index_entries(repo_path):
    """Return the number of entries in the git index."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().splitlines()
    return len(lines) if lines != [""] else 0


def _install_hook(main_repo_path, hook_script):
    """Install a pre-commit hook in the main repo (shared with worktrees)."""
    hooks_dir = main_repo_path / ".git" / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    hook_path = hooks_dir / "pre-commit"
    hook_path.write_text(hook_script)
    hook_path.chmod(0o755)


def _stage_and_commit(wt_dir):
    """Stage a new file and attempt git commit. Returns CompletedProcess."""
    (wt_dir / "new_file.txt").write_text("trigger commit\n")
    subprocess.run(["git", "add", "new_file.txt"], cwd=str(wt_dir))
    return subprocess.run(
        ["git", "commit", "-m", "test commit"],
        cwd=str(wt_dir),
        capture_output=True,
        text=True,
        timeout=300,
    )


def _init_plumb(repo_path):
    """Initialize plumb in a repo programmatically (same as plumb init)."""
    ensure_plumb_dir(repo_path)
    (repo_path / ".plumb" / "decisions").mkdir(exist_ok=True)

    spec = repo_path / "spec.md"
    spec.write_text("# Spec\n\n## Features\n\nThe system must do X.\n")

    tests_dir = repo_path / "tests"
    tests_dir.mkdir(exist_ok=True)

    cfg = PlumbConfig(
        spec_paths=["spec.md"],
        test_paths=["tests/"],
        initialized_at=datetime.now(timezone.utc).isoformat(),
    )
    save_config(repo_path, cfg)

    # Install the real plumb pre-commit hook (same string as cli.py)
    hooks_dir = repo_path / ".git" / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    hook_path = hooks_dir / "pre-commit"
    hook_path.write_text(
        '#!/bin/sh\n[ "$PLUMB_SKIP" = "1" ] && exit 0\nplumb hook\nexit $?\n'
    )
    hook_path.chmod(0o755)


@pytest.mark.slow
@needs_claude_cli
class TestClaudePWorktreeIndex:
    """Test A: claude -p called directly from a pre-commit hook."""

    def test_commit_succeeds_with_index_intact(self, tmp_path):
        """git commit in a worktree must succeed with index intact when
        claude -p is called from the pre-commit hook."""
        main_dir, wt_dir = _create_repo_with_worktree(tmp_path)
        baseline = _count_index_entries(wt_dir)
        assert baseline == 20

        _install_hook(
            main_dir,
            '#!/bin/sh\necho "say hello" | claude -p --output-format text >/dev/null 2>&1\nexit 0\n',
        )

        result = _stage_and_commit(wt_dir)
        after = _count_index_entries(wt_dir)

        assert result.returncode == 0, (
            f"Commit failed: {result.stderr[:300]}"
        )
        assert after == baseline + 1, (
            f"Expected {baseline + 1} index entries (original + new file), got {after}"
        )


@pytest.mark.slow
@needs_claude_cli
class TestPlumbHookWorktreeIndex:
    """Test B: plumb hook called from a pre-commit hook (real code path)."""

    def test_commit_succeeds_with_index_intact(self, tmp_path):
        """git commit in a worktree must succeed with index intact when
        plumb hook calls _call_claude() during pre-commit."""
        main_dir, wt_dir = _create_repo_with_worktree(tmp_path)

        # plumb init needs to happen in the main repo (worktree shares hooks)
        _init_plumb(main_dir)

        # Commit plumb files so they appear in the worktree
        repo = Repo(main_dir)
        repo.index.add([
            ".plumb/config.json",
            "spec.md",
        ])
        repo.index.commit("add plumb config")

        # Pull the new commit into the worktree
        wt_repo = Repo(wt_dir)
        wt_repo.git.merge("main", "--no-edit")

        baseline = _count_index_entries(wt_dir)

        result = _stage_and_commit(wt_dir)
        after = _count_index_entries(wt_dir)

        assert after == baseline + 1, (
            f"Expected {baseline + 1} index entries (original + new file), got {after}. "
            f"rc={result.returncode}, stderr={result.stderr[:300]}"
        )
        # plumb hook may return non-zero (pending decisions), but the index must not be corrupted
        assert "Error building trees" not in result.stderr, (
            f"Index was corrupted: {result.stderr[:300]}"
        )
