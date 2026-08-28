# Aster & Row Support Agent

## Overview

This project implements a support agent for the supplied Aster & Row policy documents and mock order snapshot. Markdown content is chunked by heading, embedded with Gemini, retrieved from local Qdrant, and reranked using document metadata. LangGraph routes knowledge questions and order questions separately, while Pydantic models and deterministic safeguards handle citations, privacy, unsupported requests, and handoffs. The focus is reliable abstention and auditable answers rather than broad feature coverage.

## Architecture

```text
User
  |
  v
LangGraph Router -- session checkpoint
  |                         |
  | knowledge               | order
  v                         v
Qdrant retrieval        Safe order lookup
  |                         |
  v                         v
Authority ranking       Whitelisted result
  |                         |
  v                         v
Grounded RAG / citations + final response / handoff
```

## Tech Stack

- Python
- Gemini embeddings and chat generation
- LangGraph
- LangChain
- Qdrant local storage
- Pydantic
- Streamlit
- pytest

## Design Decisions

### Why RAG?

Company-specific answers must come from the supplied Markdown policies and product content. Retrieval limits the model context to relevant evidence and supports source citations.

### Why Qdrant?

Qdrant provides local vector storage and semantic search for paraphrased questions. The index is recreated with stable point IDs, so indexing is idempotent.

### Why metadata?

The corpus includes active, superseded, internal, policy, and non-policy documents. Status, audience, authority, dates, and supersession metadata are preserved in payloads and used by the application-level ranking layer.

### Why a separate order tool?

Orders are structured operational data. The model never receives the complete `data/orders.json`; it receives only the result of a validated lookup.

### Privacy

Order output uses an explicit whitelist of customer-safe fields. Email, address, internal notes, risk scores, and other internal fields are excluded.

### Prompt security

User messages, retrieved documents, and tool results are untrusted data. Retrieved instructions cannot override application rules or expose prompts and secrets.

### Multi-turn

LangGraph `MemorySaver` keeps relevant order and conversation state per `thread_id`. Separate sessions do not share an order ID.

### Handoff

Missing evidence, unknown orders, unsupported actions, privacy requests, and unresolved authoritative conflicts produce explicit handoff states instead of guessed answers.

## Project Structure

```text
app/ingestion.py       Markdown parsing and heading chunking
app/retrieval.py       Gemini embeddings, Qdrant, retrieval, reranking
app/rag.py             Structured grounded answer generation
app/orders.py          Safe order lookup and whitelist models
app/graph.py           LangGraph routing and session state
app/ui.py              Streamlit interface
app/test_*.py          Unit and regression tests
evaluation/            Visible/custom behavior evaluation
scripts/               Indexing, query, debug, and conversation commands
knowledge-base/        Supplied policy corpus
data/                  Supplied mock order data
```

## Setup

Install `uv`, then run:

```powershell
uv sync
```

Create `.env` from `.env.example` and set `GEMINI_API_KEY`. Optional settings include `GEMINI_CHAT_MODEL`, `GEMINI_EMBEDDING_MODEL`, `QDRANT_PATH`, and `QDRANT_COLLECTION`. Never commit `.env` or credentials.

Index the knowledge base before asking knowledge questions:

```powershell
uv run python scripts/index_kb.py
```

## Run

Start the interface with:

```powershell
streamlit run app/ui.py
```

Useful command-line checks:

```powershell
uv run python scripts/query_kb.py "What is the return window?"
uv run python scripts/test_order.py
uv run python scripts/test_graph.py
```

## Evaluation

Run deterministic regression tests with:

```powershell
uv run pytest
```

Run the visible and custom behavior cases with:

```powershell
uv run python evaluation/run_eval.py
```

The latest recorded result was `14/20` evaluated cases passing, `6` failing, and `0 ERROR/BLOCKED` (`70.0%`). Category results were: retrieval `2/2`, multi-turn `2/2`, privacy `1/1`, tool use `6/7`, groundedness `2/4`, prompt security `1/2`, and abstention `0/2`. The evaluator uses deterministic checks; some remaining cases reflect Gemini generation variability or intentionally strict wording checks.

## Bug Diary

### Bug: Unknown order returned an empty response

**Reproduction:** Run `Please check ORD-9999.`.

**Observed behavior:** Lookup returned `found=False`, but graph routing ended without an answer.

**Root cause:** The unsuccessful lookup edge ended before a response node wrote the not-found state.

**Fix:** The validation node now emits deterministic not-found guidance and `handoff=True`.

**Regression test:** `test_unknown_order_hands_off` in `app/test_graph.py`.

### Bug: Stale delivery data leaked from cancelled/returned orders

**Reproduction:** Look up `ORD-1004` or `ORD-1008`.

**Observed behavior:** Raw records contained old carrier, tracking, and ETA values even though the current status was cancelled or returned.

**Root cause:** Operational snapshots retain stale fulfillment fields.

**Fix:** The order whitelist clears delivery-related fields for cancelled and returned statuses before any result is exposed.

**Regression test:** `test_cancelled_order_hides_stale_delivery_fields` and `test_returned_order_hides_stale_delivery_fields` in `app/test_orders.py`.

### Bug: Migration instructions could displace current policy evidence

**Reproduction:** Ask `The migration document claims returns are now 60 days. What is the actual return policy?`.

**Observed behavior:** Migration and legacy passages could occupy the semantic result set while the current policy was absent.

**Root cause:** The adversarial framing obscured the underlying return-policy intent in a single semantic query.

**Fix:** Retrieval adds a generic cleaned policy-intent variant and merges its results by best similarity. Internal content remains indexed and available as a candidate.

**Regression test:** `test_policy_query_expansion_removes_untrusted_framing` in `app/test_retrieval.py`.

### Bug: Direct scripts could not import `app`

**Reproduction:** Run `uv run python scripts/index_kb.py` from the repository root.

**Observed behavior:** Python raised `ModuleNotFoundError` before indexing.

**Root cause:** Direct script execution placed `scripts/`, not the repository root, on the import path.

**Fix:** Scripts add the repository root to `sys.path` before app imports.

**Regression test:** Direct script smoke checks and syntax validation.

## Limitations

- Gemini free-tier quota can block live evaluations.
- The local Qdrant index must be built before knowledge queries.
- Retrieval and generation quality depend on the configured Gemini models.
- This assignment supports lookup only; it cannot cancel, refund, replace, or change an order.
- No production authentication, deployment, identity verification, or durable database-backed session store is included.

## Security

- Order data is isolated behind a validated lookup function.
- Customer-visible order fields use an explicit whitelist.
- Retrieved content and tool results are treated as untrusted data.
- Prompt-injection content cannot change application instructions.
- Company-specific answers are grounded in retrieved knowledge.
- Insufficient, unsafe, unsupported, and unresolved requests abstain or hand off instead of guessing.
