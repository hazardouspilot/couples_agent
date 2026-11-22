"""
Prometheus metrics: "is the system healthy?" (LangSmith traces answer the
other question, "why did this one run do that?").

How they're collected: ONE LangChain callback handler (MetricsCallback),
passed in the run config next to the thread_id. LangChain calls its hooks as
every node, model call and tool call starts and ends, the same mechanism
LangSmith's tracer uses, so no node had to change to be measured. A few
things callbacks can't see (which safety layer fired, the turn's route,
empty searches) are counted where they happen.

Labels never contain message text or user identity: only node names,
categories, model ids and tool names. High-cardinality or personal labels
are a privacy leak AND a Prometheus performance problem.

Served by prometheus_client's start_http_server on METRICS_PORT (default
8000) at /metrics; Prometheus scrapes it (see ops/prometheus.yml).
"""

import os
from time import perf_counter

from langchain_core.callbacks import BaseCallbackHandler
from prometheus_client import Counter, Histogram, start_http_server

TURNS = Counter("agent_turns_total", "Graph turns, by route", ["route"])
NODE_LATENCY = Histogram(
    "agent_node_latency_seconds", "Time spent in each graph node", ["node"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
)
ERRORS = Counter("agent_errors_total", "Exceptions raised inside a graph node", ["node"])
LLM_REQUESTS = Counter("llm_requests_total", "Model API calls, by outcome", ["outcome"])
TOKENS = Counter("llm_tokens_total", "Model tokens", ["direction", "model"])
COST_AUD = Counter("llm_cost_aud_total", "Estimated model spend in AUD", ["model"])
SAFETY_TRIGGERS = Counter("safety_triggers_total", "Safety flags, by category and the layer "
                          "that raised them", ["category", "layer"])
TOOL_CALLS = Counter("tool_calls_total", "Memory tool calls", ["tool"])
RETRIEVAL_EMPTY = Counter("retrieval_empty_total", "search_memory calls that found nothing")

# USD per million tokens (input, output), from the Claude pricing page on
# 22 Nov 2026. Thinking tokens bill as output. Prompt-cache discounts are
# ignored, so this slightly OVER-estimates: fine for a budget alert.
PRICES_USD_PER_MTOK = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


def _aud_per_usd() -> float:
    return float(os.environ.get("AUD_PER_USD", "1.55"))  # set the current rate in .env


def estimate_cost_aud(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES_USD_PER_MTOK.get(model, (0.0, 0.0))
    usd = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    return usd * _aud_per_usd()


def turn_route(nodes_run: set[str]) -> str:
    """Which path a turn took, from the node names stream_mode='updates' saw."""
    if "crisis" in nodes_run:
        return "crisis"
    if "exercise" in nodes_run:
        return "exercise"
    return "converse"


class MetricsCallback(BaseCallbackHandler):
    """LangChain callback hooks -> Prometheus. One instance per process."""

    def __init__(self):
        self._node_starts: dict = {}  # run_id -> (node, start time)

    # A graph node is a chain run whose name equals its langgraph_node
    # metadata; nested runnables inside a node share the metadata but not the name.
    def on_chain_start(self, serialized, inputs, *, run_id, metadata=None, **kwargs):
        node = (metadata or {}).get("langgraph_node")
        if node and kwargs.get("name") == node:
            self._node_starts[run_id] = (node, perf_counter())

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        if run_id in self._node_starts:
            node, start = self._node_starts.pop(run_id)
            NODE_LATENCY.labels(node).observe(perf_counter() - start)

    def on_chain_error(self, error, *, run_id, **kwargs):
        if run_id in self._node_starts:
            node, _ = self._node_starts.pop(run_id)
            ERRORS.labels(node).inc()

    def on_llm_end(self, response, **kwargs):
        LLM_REQUESTS.labels("ok").inc()
        for generations in response.generations:
            for gen in generations:
                message = getattr(gen, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if not usage:
                    continue  # fake/offline models report no usage
                model = message.response_metadata.get("model_name", "unknown")
                tokens_in, tokens_out = usage["input_tokens"], usage["output_tokens"]
                TOKENS.labels("input", model).inc(tokens_in)
                TOKENS.labels("output", model).inc(tokens_out)
                COST_AUD.labels(model).inc(estimate_cost_aud(model, tokens_in, tokens_out))

    def on_llm_error(self, error, **kwargs):
        LLM_REQUESTS.labels("error").inc()

    def on_tool_start(self, serialized, input_str, **kwargs):
        TOOL_CALLS.labels(kwargs.get("name") or (serialized or {}).get("name", "unknown")).inc()


def start_metrics_server() -> int | None:
    """Serve /metrics. Returns the port, or None if it couldn't bind (e.g.
    a second CLI already holds it); the app works the same either way."""
    port = int(os.environ.get("METRICS_PORT", "8000"))
    try:
        start_http_server(port)
    except OSError:
        return None
    return port
