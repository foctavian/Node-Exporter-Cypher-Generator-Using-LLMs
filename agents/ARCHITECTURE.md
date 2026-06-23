# Agent Architecture

End-to-end flow: heterogeneous ESP board telemetry (semi-structured JSON) →
inferred Neo4j knowledge graph, with an LLM judge gating quality.

## Guiding principles

1. **One responsibility per node.** Each step does a single transform so it can
   be traced, evaluated, and swapped independently.
2. **LLM only where inference happens.** Acquisition and validation are
   deterministic (plain tools/functions). Reserve LLM calls for
   extraction/inference/generation and judging — keeps cost and rate-limit
   (429) pressure down.
3. **Mix agents and chains freely.** Both are `Runnable`s; a graph node is just
   `state -> state_update` and can call either. Normalize each runnable's output
   into the shared `State` at the node boundary.
4. **Keep the judge separate from the workers.** A generator must never grade
   its own output.
5. **Graph orchestration, not a free-form supervisor.** The flow is largely
   linear with a reflection loop; an explicit `StateGraph` stays deterministic
   and debuggable (per-node spans in LangSmith).

## Components

### Agents (need a decide-and-act loop)

- **Fetcher agent** — `create_agent` with the MQTT tools
  (`discover_boards`, `query_espboard_telemetry`, `query_all_boards_telemetry`).
  Used only when data is needed. Agent because it loops: discover, then query
  per board.
- **Judge agent** — evaluates the generated graph/Cypher against the source
  telemetry and existing schema. May query Neo4j to verify claims (hence an
  agent). **Must return a structured verdict** (score + issues) via a
  `submit_verdict` tool or a structured `response_format`, so the gate reads
  `state["verdict"]["score"]` rather than parsing prose.

### Chains (fixed input → structured output transforms)

- **Extractor chain** — raw board JSON → candidate entities + properties +
  units. The core of the dynamic-schema inference; where board-to-board
  heterogeneity becomes structure.
- **Relationship-inference chain** — entities + existing schema → `HAS_*` edges.
- **Cypher-generation chain** — entities + relationships → `MERGE/SET` scripts.

### Deterministic (tools/functions, no LLM)

- **Acquisition** — the MQTT layer (pure I/O).
- **Validation/execution** — Cypher syntax check, schema-existence checks,
  running against Neo4j (reuse `cypher_validation`, `_batch_node_exists`).

## Orchestration (StateGraph)

```
        ┌─────────┐
START ─▶ │  fetch  │  (agent, conditional: only if data needed)
        └────┬────┘
             ▼
        ┌─────────┐
        │ extract │  (chain)
        └────┬────┘
             ▼
        ┌──────────────┐
        │ infer_rels   │  (chain)
        └────┬─────────┘
             ▼
        ┌──────────────┐
        │ gen_cypher   │  (chain)
        └────┬─────────┘
             ▼
        ┌──────────────┐
        │  validate    │  (deterministic: syntax + schema + execute)
        └────┬─────────┘
             ▼
        ┌──────────────┐
        │    judge     │  (agent → structured verdict)
        └────┬─────────┘
             │ accept ──────────▶ commit ─▶ END
             │ reject ──▶ back to the offending stage with the
             │            judge's feedback (reflection loop)
```

- Deterministic edges for the happy path.
- Conditional edge after `judge`: accept → commit; reject → route back to the
  stage the verdict blames, passing the judge feedback as input. This is the
  existing `reflect` loop, driven by a quality judge instead of only syntax
  errors.

## Boundary contract

Define a single `State` TypedDict as the canonical contract. Each node:
1. reads its inputs from `State`,
2. calls its agent or chain,
3. normalizes the output (extract the final message content for an agent;
   take the structured object for a chain) back into `State`.

```python
class State(TypedDict):
    fetch_request: str
    telemetry: str            # filled by fetch (agent)
    entities: ...             # filled by extract (chain)
    relationships: ...        # filled by infer_rels (chain)
    script: SystemCypherScript
    verdict: dict             # {"score": float, "issues": [...]} from judge
    error: str
    iterations: int
```

Agent node vs chain node differ only in the body:

```python
def fetch_node(state: State) -> State:                      # agent
    out = fetcher_agent.invoke({"messages": [{"role": "user", "content": state["fetch_request"]}]})
    return {**state, "telemetry": out["messages"][-1].content}

def extract_node(state: State) -> State:                    # chain
    return {**state, "entities": extraction_chain.invoke({"telemetry": state["telemetry"]})}
```

## Open decisions

- Judge scope: start with **one end-to-end judge** + deterministic per-stage
  validators. Add per-stage judges only if fidelity demands it (multiplies LLM
  calls).
- Whether `fetch` runs is conditional on whether telemetry is already in state.
- Data format: feed **raw JSON** to the extractor (preserves nesting/units) so
  heterogeneous boards drive genuine schema inference; the `telemetry_to_exporter`
  converter is kept only for external Prometheus/node_exporter compatibility.
