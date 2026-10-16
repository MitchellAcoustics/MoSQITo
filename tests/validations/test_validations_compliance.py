"""Assert the pass/fail verdicts that the validation scripts already compute.

Each case uses the script's own reference data and tolerance (ISO 532-1,
DIN 45692, ANSI S3.5, Zwicker and Fastl, ECMA 418-2 / ArtemiS). The test only
turns the verdict the script draws on its plot into an assertion.

Cases that do not comply are listed in KNOWN_FAILURES and marked
``xfail(strict=True)``. A change that makes one of them pass therefore fails
the suite until the entry is removed, so this list records which cases a
change fixed or broke.
"""

import pytest

from ._scripts import VALIDATIONS, load_script

ZWST = "sq_metrics/loudness_zwst/validation_loudness_zwst.py"
ZWTV = "sq_metrics/loudness_zwtv/validation_loudness_zwtv.py"
SHARPNESS = "sq_metrics/sharpness_din/validation_sharpness_din.py"
SII = "sq_metrics/speech_intelligibility/validation_sii.py"
ROUGHNESS_DW = "sq_metrics/roughness_dw/validation_roughness_danielweber.py"
ROUGHNESS_ECMA = "sq_metrics/roughness_ecma/validation_roughness_ecma.py"
SPECIFIC_ROUGHNESS_ECMA = "sq_metrics/roughness_ecma/validation_specific_roughness_ecma.py"

# Case id -> why it does not comply.
KNOWN_FAILURES = {
    "roughness_dw-fc2000": (
        "above the Zwicker and Fastl curve by more than 0.1 asper for "
        "fmod 80-140 Hz (up to 0.17 asper at 120 Hz)"
    ),
    **{
        f"specific_roughness_ecma-fc1000-fm{fm}": (
            "specific roughness above the ArtemiS reference by more than 10% "
            "in about 10 of the 53 bands (up to 0.008 asper/Bark)"
        )
        for fm in (40, 70, 120)
    },
}


def case(case_id, *values, marks=()):
    marks = list(marks)
    if case_id in KNOWN_FAILURES:
        marks.append(pytest.mark.xfail(strict=True, reason=KNOWN_FAILURES[case_id]))
    return pytest.param(*values, id=case_id, marks=marks)


# --- ISO 532-1 stationary loudness (annex B2 and B3) -------------------------

ZWST_SIGNALS = [
    "test_signal_1",
    "Test signal 2 (250 Hz 80 dB)",
    "Test signal 3 (1 kHz 60 dB)",
    "Test signal 4 (4 kHz 40 dB)",
    "Test signal 5 (pinknoise 60 dB)",
]


@pytest.fixture(scope="module")
def zwst_results():
    script = load_script(ZWST)
    results = {
        **script["validation_loudness_zwst_3oct"](),
        **script["validation_loudness_zwst"](),
    }
    assert sorted(results) == sorted(ZWST_SIGNALS)
    return results


@pytest.mark.parametrize(
    "signal, quantity",
    [
        case(f"zwst-{signal}-{quantity}", signal, quantity)
        for signal in ZWST_SIGNALS
        for quantity in ("N", "N_specific")
    ],
)
def test_loudness_zwst(zwst_results, signal, quantity):
    is_N_close, is_N_specific_close = zwst_results[signal]
    assert {"N": is_N_close, "N_specific": is_N_specific_close}[quantity]


# --- ISO 532-1 time-varying loudness (annex B4 and B5) -----------------------


def _zwtv_cases():
    signals = load_script(ZWTV)["signal"]
    return [
        case(f"zwtv-{s['tab']}", i, marks=[pytest.mark.slow])
        for i, s in enumerate(signals)
    ]


@pytest.mark.parametrize("index", _zwtv_cases())
def test_loudness_zwtv(index):
    script = load_script(ZWTV)
    assert script["validation_loudness_zwtv"](script["signal"][index])


# --- DIN 45692 sharpness (chapter 6) ------------------------------------------


@pytest.fixture(scope="module")
def sharpness_results():
    script = load_script(SHARPNESS)
    return {
        noise_type: script["validation_sharpness_din"](script[noise_type])
        for noise_type in ("broadband", "narrowband")
    }


def _sharpness_cases():
    script = load_script(SHARPNESS)
    return [
        case(f"sharpness-{noise['data_file'].split('/')[-1].split('.')[0]}", noise_type, i)
        for noise_type in ("broadband", "narrowband")
        for i, noise in enumerate(script[noise_type])
    ]


@pytest.mark.parametrize("noise_type, index", _sharpness_cases())
def test_sharpness_din(sharpness_results, noise_type, index):
    assert sharpness_results[noise_type][index]


# --- ANSI S3.5-1997 speech intelligibility index -----------------------------


@pytest.mark.parametrize("procedure", [case("sii-octave", "octave")])
def test_sii_ansi(procedure):
    # The third-octave check in the script is not asserted: it recomputes the
    # equivalent noise spectrum in the script itself instead of calling MoSQITo.
    script = load_script(SII)
    assert script["validation_sii_octave"](script["reference"][0])


# --- Daniel and Weber roughness, Zwicker and Fastl reference -----------------

ROUGHNESS_CARRIERS = [125, 250, 500, 1000, 2000, 4000, 8000]


@pytest.mark.parametrize(
    "fc", [case(f"roughness_dw-fc{fc}", fc) for fc in ROUGHNESS_CARRIERS]
)
def test_roughness_dw(fc):
    assert load_script(ROUGHNESS_DW)["validation_roughness_dw"](fc)


# --- ECMA 418-2 roughness, Zwicker and Fastl reference -----------------------


@pytest.mark.parametrize(
    "index",
    [
        case(f"roughness_ecma-fc{fc}", i, marks=[pytest.mark.slow])
        for i, fc in enumerate(ROUGHNESS_CARRIERS)
    ],
)
def test_roughness_ecma(index):
    script = load_script(ROUGHNESS_ECMA)
    assert list(script["fc_vector"]) == ROUGHNESS_CARRIERS
    fc = script["fc_vector"][index]
    assert script["validation_roughness_ecma"](fc, script["fm_vector_for"](index))


# --- ECMA 418-2 specific roughness, ArtemiS reference ------------------------


@pytest.mark.parametrize(
    "fmod", [case(f"specific_roughness_ecma-fc1000-fm{fm}", fm) for fm in (40, 70, 120)]
)
def test_specific_roughness_ecma(fmod):
    script = load_script(SPECIFIC_ROUGHNESS_ECMA)
    ref_file = VALIDATIONS / "sq_metrics/roughness_ecma/input/validation_specific_roughness_ecma.xlsx"
    assert script["validation_specific_roughness_ecma"](1000, fmod, str(ref_file))


def test_known_failures_name_real_cases():
    """Guard against stale or misspelt ids in KNOWN_FAILURES."""
    case_ids = {
        param.id
        for test in (test_loudness_zwst, test_loudness_zwtv, test_sharpness_din,
                     test_sii_ansi, test_roughness_dw, test_roughness_ecma,
                     test_specific_roughness_ecma)
        for mark in test.pytestmark
        if mark.name == "parametrize"
        for param in mark.args[1]
    }
    assert set(KNOWN_FAILURES) <= case_ids, set(KNOWN_FAILURES) - case_ids
