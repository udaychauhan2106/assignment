from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.run_eval import _concept_present, _load_cases, _normalize
from evaluation.run_eval import _phrase_present


def test_all_visible_cases_are_included_without_replacing_them() -> None:
    cases = _load_cases("visible-cases.json")

    assert len(cases) >= 14
    assert {case["id"] for case in cases} == {
        "standard-return-window",
        "trailplus-return-window",
        "final-sale-damaged-exception",
        "canada-multiturn",
        "unsupported-country",
        "valid-order-lookup",
        "missing-order-id",
        "cancelled-order-stale-eta",
        "unknown-order",
        "shipped-without-eta",
        "order-data-privacy",
        "no-lifetime-warranty",
        "retrieved-prompt-injection",
        "insufficient-information",
        "genuine-active-source-conflict",
    }


def test_at_least_five_custom_cases_are_unique() -> None:
    visible_ids = {case["id"] for case in _load_cases("visible-cases.json")}
    custom_cases = _load_cases("custom-cases.json")
    custom_ids = {case["id"] for case in custom_cases}

    assert len(custom_cases) >= 5
    assert custom_ids.isdisjoint(visible_ids)


def test_concept_matching_is_tolerant_of_common_wording() -> None:
    assert _concept_present(
        "5–9 business days after dispatch",
        "Canadian orders arrive in 5-9 business days after dispatch.",
    )
    assert _concept_present(
        "no lifetime warranty",
        "Aster & Row does not offer a lifetime warranty.",
    )
    assert _normalize("A 5–9 day estimate") == "a 5 9 day estimate"
    assert _phrase_present("45 calendar days", "a 45-calendar-day return window")
    assert _phrase_present("5–9 business days", "5-9 business days after dispatch")
    assert _concept_present(
        "human confirmation or safest interim guidance",
        "Please confirm this with a human support representative.",
    )
