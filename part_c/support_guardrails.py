"""Part C: privacy-safe, resilient support ticket triage using the supplied mock_llm."""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mock_llm  # noqa: E402 -- supplied file; do not modify

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
ALLOWED = {"billing", "delivery", "technical", "fraud"}
PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]
ZERO_SHOT = (
    'CLASSIFY the support ticket as exactly one of: billing, delivery, technical, fraud. '
    'Reply JSON only in the form {"category": "...", "reason": "..."}.'
)
FEW_SHOT = (
    ZERO_SHOT + '\nEXAMPLES:\n'
    'Ticket: Someone hacked my account. -> {"category":"fraud","reason":"unauthorized access"}\n'
    'Ticket: My delivery arrived late. -> {"category":"delivery","reason":"late delivery"}\n'
    'Ticket: The app crashes. -> {"category":"technical","reason":"app error"}'
)


def redact(text):
    counts = {name: 0 for name, _, _ in PATTERNS}
    for name, pattern, replacement in PATTERNS:
        text, count = pattern.subn(replacement, text)
        counts[name] += count
    assert all(not pattern.search(text) for _, pattern, _ in PATTERNS), "PII remains"
    return text, counts


def retry_chat(ticket_id, messages):
    for retry_number in range(4):  # initial attempt plus at most 3 retries
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_number == 3:
                return None
            wait = BASE_DELAY * 2 ** retry_number * random.uniform(0.9, 1.15)
            print(f"RETRY ticket={ticket_id} retry={retry_number} wait={wait:.5f}s")
            time.sleep(wait)
    return None


def validate_reply(content):
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        words = set(re.findall(r"[a-z]+", (content or "").lower()))
        matched = words & ALLOWED
        if len(matched) == 1:
            return "classified_keyword", matched.pop()
        return "flagged_invalid_output", None
    if isinstance(data, dict) and data.get("category") in ALLOWED:
        return "classified_json", data["category"]
    return "flagged_invalid_output", None


def is_premium_value(value):
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def load_segments():
    path = ROOT / "outputs" / "customer_segments.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run Part B first: {path}")
    frame = pd.read_csv(path, dtype={"customer_id": str})
    return {
        row.customer_id: (str(row.segment_name), is_premium_value(row.is_premium))
        for row in frame.itertuples(index=False)
    }


def route_ticket(status, category, premium):
    if status not in {"classified_json", "classified_keyword"}:
        return "NORMAL", "human_review"
    if category == "fraud":
        return "HIGH", "security_team"
    if premium:
        return "HIGH", "premium_desk"
    return "NORMAL", "support_queue"


def run_pipeline(tickets, segments, prompt, budget=None):
    mock_llm.reset()  # independent, reproducible run
    records = []
    usage_total = 0
    redaction_counts = {name: 0 for name, _, _ in PATTERNS}
    for ticket in tickets.itertuples(index=False):
        raw = "" if pd.isna(ticket.text) else str(ticket.text)
        redacted, counts = redact(raw)
        for key, count in counts.items():
            redaction_counts[key] += count
        category = None
        tokens_used = 0
        if not raw.strip():
            status = "skipped_empty"
        elif mock_llm.est_tokens(raw) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        elif budget is not None and usage_total >= budget:
            status = "budget_exhausted"
        else:
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": redacted}]
            # No raw email, phone number, or order ID may reach the model.
            assert all(not p.search(m["content"]) for m in messages for _, p, _ in PATTERNS)
            assert messages[-1]["content"] == redacted
            response = retry_chat(ticket.ticket_id, messages)
            if response is None:
                status = "flagged_invalid_output"  # fail closed after exhausted retries
            else:
                tokens_used = response.usage.total_tokens
                usage_total += tokens_used
                status, category = validate_reply(response.choices[0].message.content)
        segment_name, premium = segments.get(str(ticket.customer_id), (None, False))
        priority, route = route_ticket(status, category, premium)
        records.append({
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used,
            "true_category": ticket.true_category,
        })
    return pd.DataFrame(records), usage_total, redaction_counts


def report(label, result, tokens, redactions):
    eligible = result[result.status.isin(["classified_json", "classified_keyword"])]
    correct = int((eligible.category == eligible.true_category).sum())
    accuracy = 100 * correct / len(eligible) if len(eligible) else 0
    print(f"\n{label}")
    print("Status counts:\n" + result.status.value_counts().to_string())
    print(f"Classified: {len(eligible)}; correct: {correct}; accuracy: {accuracy:.2f}%")
    print(f"Total tokens: {tokens}; tokens per classified ticket: {tokens / len(eligible):.2f}" if len(eligible) else f"Total tokens: {tokens}")
    print("PII substitutions:", redactions, "total:", sum(redactions.values()))
    return accuracy


def main():
    tickets = pd.read_csv(ROOT / "data" / "raw" / "tickets.csv", dtype={"ticket_id": str, "customer_id": str})
    segments = load_segments()
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)

    zero, zero_tokens, redactions = run_pipeline(tickets, segments, ZERO_SHOT)
    zero_accuracy = report("ZERO SHOT (no budget)", zero, zero_tokens, redactions)
    print("Zero-shot confusion table (true rows, predicted columns):")
    classified = zero[zero.status.isin(["classified_json", "classified_keyword"])]
    print(pd.crosstab(classified.true_category, classified.category).to_string())

    few, few_tokens, few_redactions = run_pipeline(tickets, segments, FEW_SHOT)
    few_accuracy = report("FEW SHOT (no budget)", few, few_tokens, few_redactions)
    assert redactions == few_redactions
    assert len(tickets) == len(few) == 164
    assert redactions == {"email": 55, "phone": 91, "order_id": 132}, redactions
    assert sum(redactions.values()) == 278
    assert (zero.status.value_counts().get("classified_json", 0),
            zero.status.value_counts().get("classified_keyword", 0),
            zero.status.value_counts().get("flagged_invalid_output", 0)) == (135, 12, 9)
    assert abs(zero_accuracy - 85.7142857) < 0.5
    assert abs(few_accuracy - 100) < 0.5

    demo_budget = 6000
    demo, demo_tokens, _ = run_pipeline(tickets, segments, ZERO_SHOT, budget=demo_budget)
    if not (demo.status == "budget_exhausted").any():
        demo_budget = 5000
        demo, demo_tokens, _ = run_pipeline(tickets, segments, ZERO_SHOT, budget=demo_budget)
    demo.drop(columns="true_category").to_csv(output / "triage_budget_demo.csv", index=False)
    print(f"\nBUDGET DEMO: budget={demo_budget}, spent={demo_tokens}, exhausted={(demo.status == 'budget_exhausted').sum()}")
    assert (demo.status == "budget_exhausted").any()

    final, final_tokens, _ = run_pipeline(tickets, segments, FEW_SHOT, budget=100000)
    final.drop(columns="true_category").to_csv(output / "triage_results.csv", index=False)
    print(f"FINAL: {len(final)} tickets, {final_tokens} tokens, route counts:")
    print(final.route.value_counts().to_string())
    print("Saved outputs/triage_budget_demo.csv and outputs/triage_results.csv")


if __name__ == "__main__":
    main()
