# Eval report 2026-11-22-1503 (holdout-v04-classifier)

Safety dataset: `safety_holdout.jsonl`. Gate: v04 regex + LLM classifier. Model: `claude-sonnet-5-5`.

## 1. Safety gate (v04 regex + LLM classifier)

40 messages: abuse 8, harm_to_others 4, none 20, self_harm 8

| Category | Support | Precision | Recall | TP | FP | FN |
|---|---|---|---|---|---|---|
| self_harm | 8 | 100% | 100% | 8 | 0 | 0 |
| harm_to_others | 4 | 100% | 100% | 4 | 0 | 0 |
| abuse | 8 | 100% | 100% | 8 | 0 | 0 |
| **any flag** | 20 | 100% | 100% | 20 | 0 | 0 |

**By kind** (flagged / n; positives in brackets):

- direct: 6/6 flagged (6 positives)
- discussing: 0/6 flagged (0 positives)
- idiom: 0/14 flagged (0 positives)
- indirect: 6/6 flagged (6 positives)
- ipv: 8/8 flagged (8 positives)

**Misses (0):**

| Text | Gold | Predicted | Kind |
|---|---|---|---|
