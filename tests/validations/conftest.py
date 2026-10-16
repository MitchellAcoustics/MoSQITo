import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


@pytest.fixture(scope="package", autouse=True)
def no_plot_output():
    """The scripts plot as they validate: never show figures and never
    overwrite the tracked PNGs in ``validations/**/output/``.

    Package scope, so that it is already active when the module-scoped
    fixtures that run whole validations are set up.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(plt, "show", lambda *args, **kwargs: None)
        mp.setattr(Figure, "savefig", lambda *args, **kwargs: None)
        yield


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")
