import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run tests marked 'slow'",
    )


def pytest_collection_modifyitems(config, items):
    """Skip 'slow' tests unless asked for with --runslow or a -m expression
    that mentions 'slow'. Unlike ``addopts = -m "not slow"``, this survives a
    user-supplied -m and says why a test was skipped."""
    if config.getoption("--runslow") or "slow" in config.getoption("markexpr", ""):
        return
    skip_slow = pytest.mark.skip(reason="slow: run with --runslow or -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
