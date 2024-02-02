from os import PathLike
from types import ModuleType

import git
from packaging import version

import gnn_tracking
from gnn_tracking.utils.log import logger


def get_commit_hash(module: None | ModuleType | str | PathLike = None) -> str:
    """Get the git commit hash of the current module."""

    if module is None:
        import gnn_tracking

        module = gnn_tracking
    base_path = module.__path__[0] if isinstance(module, ModuleType) else module
    try:
        repo = git.Repo(path=base_path, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        logger.warning(
            "Could not find git repository at %s. This happens if you "
            "don't use an editable install.",
            base_path,
        )
        return "invalid"
    if repo.is_dirty():
        logger.warning(
            "Repository %s is dirty, commit hash may not be accurate.", base_path
        )
    return repo.head.object.hexsha


def assert_version_geq(require: str):
    assert version.parse(gnn_tracking.__version__) >= version.parse(require), (
        f"Please update gnn_tracking from {gnn_tracking.__version__} to at least "
        f"version {require}."
    )
