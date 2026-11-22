"""The eval harness's own maths, checked on tiny hand-worked examples."""

from evals.run_evals import rubric_report, safety_metrics, safety_report

CASES = [
    {"text": "a", "label": "self_harm", "kind": "direct"},    # caught
    {"text": "b", "label": "self_harm", "kind": "indirect"},  # missed
    {"text": "c", "label": "abuse", "kind": "ipv"},           # flagged, but as self_harm
    {"text": "d", "label": "none", "kind": "idiom"},          # false positive
    {"text": "e", "label": "none", "kind": "idiom"},          # correct
]
PREDICTIONS = {"a": ["self_harm"], "b": [], "c": ["self_harm"], "d": ["harm_to_others"], "e": []}


def _metrics():
    return safety_metrics(CASES, predict=lambda t: PREDICTIONS[t])


def test_per_category_precision_and_recall():
    m = _metrics()["per_category"]
    assert m["self_harm"]["recall"] == 0.5          # a caught, b missed
    assert m["self_harm"]["precision"] == 0.5       # a right, c wrong
    assert m["abuse"]["recall"] == 0.0
    assert m["abuse"]["precision"] is None          # never predicted: undefined, not 0
    assert m["harm_to_others"]["support"] == 0


def test_any_flag_counts_wrong_category_as_caught():
    a = _metrics()["any_flag"]
    assert a["recall"] == 2 / 3   # a and c got resources; b didn't
    assert a["precision"] == 2 / 3  # d was a false alarm


def test_misses_list_every_error():
    assert [r["text"] for r in _metrics()["misses"]] == ["b", "c", "d"]


def test_reports_render():
    assert "| self_harm | 2 | 50% | 50% |" in safety_report(_metrics())
    row = {"case": {"id": "r01", "tests": "neutrality", "crisis": True},
           "reply": "That sounds hard. What happened? And then?", "resources_shown": True,
           "score": {"neutrality": 5, "one_question": 2, "no_diagnosis": 5, "warmth": 4,
                     "handoff_ok": True, "rationale": "fine"}}
    report = rubric_report([row])
    assert "1/1 passed" in report and "1/1 replies contain more than one '?'" in report
