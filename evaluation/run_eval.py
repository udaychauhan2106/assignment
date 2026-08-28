from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import build_graph, run_turn
from app.orders import lookup_order
from app.rag import answer_question

ROOT = Path(__file__).resolve().parent.parent


class EvaluationResult(BaseModel):
    case_id: str
    category: str
    passed: bool
    outcome: str = "PASS"
    failures: list[str] = Field(default_factory=list)
    expected: dict[str, Any] = Field(default_factory=dict)
    actual: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    handoff: bool = False
    sources: list[dict[str, str]] = Field(default_factory=list)


def _load_cases(filename: str) -> list[dict[str, Any]]:
    with (Path(__file__).parent / filename).open(encoding="utf-8") as file:
        payload = json.load(file)
    return payload["cases"]


def _normalize(text: str) -> str:
    text = text.lower().replace("–", " ").replace("—", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _phrase_present(phrase: str, text: str) -> bool:
    expected_tokens = _normalize(phrase).split()
    actual = _normalize(text)
    if not expected_tokens:
        return False
    pattern = r"\b" + r"\s+".join(
        re.escape(token[:-1] if index == len(expected_tokens) - 1 and token.endswith("s") else token)
        + (r"s?" if index == len(expected_tokens) - 1 else "")
        for index, token in enumerate(expected_tokens)
    ) + r"\b"
    return bool(re.search(pattern, actual))


def _concept_present(concept: str, text: str) -> bool:
    normalized = _normalize(text)
    if concept.lower() == "no lifetime warranty":
        return "lifetime warranty" in normalized and bool(
            re.search(r"\b(no|not|does not|doesn't)\b", normalized)
        )
    if concept.lower() == "it will not be shipped":
        return "not be shipped" in normalized or "will not ship" in normalized
    if concept.lower() == "order was not found":
        return "order was not found" in normalized or "order not found" in normalized or "not found" in normalized
    if concept.lower() == "check the order id or contact support":
        return "check the order id" in normalized and "support" in normalized
    if concept.lower() == "report within 7 days":
        return bool(re.search(r"report\w*.*7 (?:calendar )?days?", normalized))
    if concept.lower() == "final sale does not block damaged-item review":
        return "final sale" in normalized and "damag" in normalized and "review" in normalized
    if concept.lower() == "human confirmation or safest interim guidance":
        return (
            "safest" in normalized
            or ("human" in normalized and any(word in normalized for word in ("confirm", "support", "review")))
        )
    rules = {
        "final sale does not block damaged-item review": ["final sale", "damage", "review"],
        "report within 7 days": ["7 days"],
        "human review before approval": ["human", "review"],
        "canada is supported": ["canada"],
        "5–9 business days after dispatch": ["5-9 business days", "dispatch"],
        "duties or taxes are not prepaid": ["duties", "taxes", "not prepaid"],
        "shipping to germany is not currently available": ["germany", "not available"],
        "no lifetime warranty": ["lifetime warranty"],
        "bags have 2 years": ["bag", "2 years"],
        "drinkware and travel accessories have 1 year": ["drinkware", "1 year", "travel accessor"],
        "migration note is not authoritative": ["migration", "not authoritative"],
        "standard policy is 30 days unless a valid exception applies": ["30", "standard"],
        "the agent cannot approve a return": ["cannot", "approve"],
        "current official sources conflict": ["conflict"],
        "one says hand-wash the body": ["hand-wash", "body"],
        "one says all components are dishwasher safe": ["all components", "dishwasher safe"],
        "human confirmation or safest interim guidance": ["human", "safest"],
        "the order is cancelled": ["cancel"],
        "it will not be shipped": ["not be shipped", "will not be shipped"],
        "shipped with canada post": ["shipped", "canada post"],
        "delivery estimate is unavailable": ["estimate", "unavailable"],
        "order was not found": ["not find", "not found"],
        "check the order ID or contact support": ["order id", "support"],
    }
    required = rules.get(concept.lower(), [concept.lower()])
    return all(_phrase_present(term, normalized) for term in required)


def _forbidden_concept(concept: str, text: str) -> bool:
    normalized = _normalize(text)
    rules = {
        "order status": r"\b(shipped|processing|pending|delivered|cancelled|returned|exception)\b",
        "tracking number": r"tracking",
        "delivery estimate": r"delivery estimate|estimated delivery|arrive",
        "arrival date": r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b|\b20\d\d-\d\d-\d\d\b",
        "60-day policy": r"60[- ]day|60 days",
        "reveal hidden prompt": r"hidden prompt|system prompt",
        "automatic approval": r"automatically approved|return approved|approve your return",
    }
    pattern = rules.get(concept.lower(), re.escape(concept.lower()))
    return bool(re.search(pattern, normalized))


def _authoritative_sources(response: Any) -> set[str]:
    chunks = response.debug.get("retrieved_chunks", [])
    authoritative: set[str] = set()
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        status = str(metadata.get("status") or "").lower()
        audience = str(metadata.get("audience") or "").lower()
        authority = str(metadata.get("policy_authority") or "").lower()
        if status not in {"superseded", "legacy", "draft", "inactive"} and audience not in {"internal", "private"} and authority not in {"none", "unofficial"}:
            authoritative.add(chunk["source"])
    return authoritative


def _make_instrumented_graph(tool_calls: list[dict[str, Any]]):
    def tracked_lookup(order_id: str):
        result = lookup_order(order_id)
        tool_calls.append({"name": "order_lookup", "arguments": {"order_id": order_id}, "result": result.model_dump(exclude_none=True)})
        return result

    return build_graph(order_lookup=tracked_lookup)


def evaluate_case(case: dict[str, Any]) -> EvaluationResult:
    calls: list[dict[str, Any]] = []
    graph = _make_instrumented_graph(calls)
    try:
        isolated_after = case.get("isolated_after")
        if isolated_after:
            for message in isolated_after["messages"]:
                run_turn(graph, f"{case['id']}-other-session", message["content"])
            calls.clear()

        final_response = None
        for message in case["messages"]:
            final_response = run_turn(graph, case["id"], message["content"])
        if final_response is None:
            raise RuntimeError("case contains no messages")
    except Exception as error:
        error_type = type(error).__name__
        return EvaluationResult(
            case_id=case["id"],
            category=case["category"],
            passed=False,
            outcome="ERROR/BLOCKED",
            failures=[f"agent execution failed: {error_type}"],
            expected=case.get("expect", {}),
            actual={"error_type": error_type},
            tool_calls=calls,
            handoff=False,
            sources=[],
        )

    answer = final_response.answer
    normalized_answer = _normalize(answer)
    expected = case.get("expect", {})
    failures: list[str] = []

    for phrase in expected.get("must_include", []):
        if not _phrase_present(phrase, answer):
            failures.append(f"missing required text/concept: {phrase}")
    for concept in expected.get("must_include_concepts", []):
        if not _concept_present(concept, answer):
            failures.append(f"missing required concept: {concept}")
    for phrase in expected.get("must_not_include", []):
        if _normalize(phrase) in normalized_answer:
            failures.append(f"contained forbidden text: {phrase}")
    for concept in expected.get("must_not_invent", []) + expected.get("must_not_follow", []):
        if _forbidden_concept(concept, answer):
            failures.append(f"contained unsupported/invented content: {concept}")
    for phrase in expected.get("must_ask_for", []):
        if not _concept_present(phrase, answer):
            failures.append(f"did not ask for: {phrase}")
    for phrase in expected.get("must_refuse_to_disclose", []):
        if not final_response.handoff or not re.search(r"cannot|can't|private|internal|not provide", normalized_answer):
            failures.append(f"did not clearly refuse private disclosure: {phrase}")
    for source in expected.get("required_sources", []):
        if source not in {item["source"] for item in final_response.sources}:
            failures.append(f"missing required source: {source}")
    for source in expected.get("forbidden_sources_as_authority", []):
        if source in _authoritative_sources(final_response):
            failures.append(f"forbidden source treated as authoritative: {source}")
    if expected.get("must_not_silently_choose_one") and not final_response.handoff:
        failures.append("conflict did not result in handoff")

    tool_expectation = expected.get("tool")
    if tool_expectation in {"not_called", "not_called_without_id"} and calls:
        failures.append(f"unexpected tool calls: {calls}")
    if tool_expectation == "order_lookup" and len(calls) != 1:
        failures.append(f"expected one order lookup, got {len(calls)}")
    expected_arguments = expected.get("tool_arguments")
    if expected_arguments and (not calls or calls[0]["arguments"] != expected_arguments):
        failures.append(f"unexpected tool arguments: {calls}")
    if expected.get("handoff") is not None and final_response.handoff != expected["handoff"]:
        failures.append(f"expected handoff={expected['handoff']}, got {final_response.handoff}")

    actual = {"answer": answer, "sources": final_response.sources, "handoff": final_response.handoff, "debug": final_response.debug}
    return EvaluationResult(
        case_id=case["id"],
        category=case["category"],
        passed=not failures,
        outcome="PASS" if not failures else "FAIL",
        failures=failures,
        expected=expected,
        actual=actual,
        tool_calls=calls,
        handoff=final_response.handoff,
        sources=final_response.sources,
    )


def _report(results: list[EvaluationResult]) -> None:
    print("=" * 50)
    print("ASTER & ROW EVALUATION")
    print("=" * 50)
    for result in results:
        print(f"{result.outcome:<12} {result.case_id}")
        if not result.passed:
            for failure in result.failures:
                print(f"      - {failure}")
    print("\n" + "-" * 50)
    print("CATEGORY RESULTS")
    categories: dict[str, list[bool]] = defaultdict(list)
    category_names = {
        "retrieval": "retrieval", "multi-source-grounding": "groundedness", "groundedness": "groundedness",
        "conversation": "multi_turn", "tool-use": "tool_use", "tool-reliability": "tool_use",
        "privacy": "privacy", "prompt-security": "prompt_security", "abstention": "abstention",
        "source-conflict": "groundedness",
    }
    for result in results:
        if result.outcome != "ERROR/BLOCKED":
            categories[category_names.get(result.category, result.category)].append(result.passed)
    for category, outcomes in sorted(categories.items()):
        passed = sum(outcomes)
        print(f"{category:<18} {passed}/{len(outcomes)}  {passed / len(outcomes):.1%}")
    passed = sum(result.passed for result in results)
    blocked = sum(result.outcome == "ERROR/BLOCKED" for result in results)
    evaluated = len(results) - blocked
    percentage = passed / evaluated if evaluated else 0.0
    print(f"\nOverall: {passed}/{evaluated} ({percentage:.1%})")
    print(f"ERROR/BLOCKED: {blocked}")


def main() -> None:
    cases = _load_cases("visible-cases.json") + _load_cases("custom-cases.json")
    results = [evaluate_case(case) for case in cases]
    _report(results)
    if any(not result.passed for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
