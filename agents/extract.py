"""Entity extraction from telemetry JSON (see agents/ARCHITECTURE.md).

The LLM does the inference: given a board's semi-structured telemetry, it
identifies the graph nodes (labels, identity keys, properties) and the
relationships between them. No hand-coded structural rules — that is the point,
so heterogeneous / previously-unseen board shapes are handled without code
changes.
"""
import json
from typing import cast
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


class CandidateNode(BaseModel):
    label: str = Field(description="Graph label in UPPER_SNAKE_CASE, e.g. BOARD, TASK, SENSOR.")
    key: dict = Field(description="Identity properties that uniquely identify this node (used for MERGE). Always include board_id; sub-entities also need a name.")
    properties: dict = Field(default_factory=dict, description="Scalar attributes of the node (numbers, strings, booleans).")


class CandidateRel(BaseModel):
    source: CandidateNode = Field(description="Parent node; only its label and key matter here.")
    target: CandidateNode = Field(description="Child node; only its label and key matter here.")
    type: str = Field(description="Relationship type in UPPER_SNAKE_CASE, e.g. HAS_TASK.")


class Extraction(BaseModel):
    nodes: list[CandidateNode]
    rels: list[CandidateRel]


_SYSTEM_PROMPT = """You extract a property graph from a single ESP board's telemetry JSON.

The telemetry is semi-structured and varies between boards (different sensors,
tasks, capabilities). Infer the graph from the data itself — do not assume a
fixed schema.

Rules:
- The top-level object is a BOARD node, identified by board_id.
- A nested object that represents a real thing (a sensor, a task, a subsystem)
  is its own node; a scalar field is a property of its enclosing node.
- An array of objects yields one node per element.
- Every node's key must include board_id; sub-entities also need a name.
- Labels and relationship types are UPPER_SNAKE_CASE. Parent-to-child
  relationships are typically HAS_<LABEL> (e.g. BOARD -[:HAS_SENSOR]-> SENSOR).
- Put scalar attributes in `properties`. In relationships, only the source and
  target label+key matter.
- Use meaningful labels for the entities you recognize (e.g. a dht22 object is a
  SENSOR), not the raw JSON key.
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("user", "Existing graph schema (may be empty):\n{schema}\n\nTelemetry:\n{telemetry}"),
])

_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        load_dotenv()
        llm = init_chat_model("mistralai:codestral-latest", temperature=0)
        _chain = _prompt | llm.with_structured_output(Extraction)
    return _chain


def extract_candidates(payload, existing_schema: str | None = None) -> tuple[list[CandidateNode], list[CandidateRel]]:
    telemetry = payload if isinstance(payload, str) else json.dumps(payload)
    result = cast(Extraction, _get_chain().invoke({
        "telemetry": telemetry,
        "schema": existing_schema or "(none)",
    }))
    return result.nodes, result.rels


if __name__ == "__main__":
    sample = {
        "type": "telemetry", "board_id": 51, "board_type": "esp32-dht",
        "free_heap_bytes": 123456, "wifi_rssi_dbm": -57,
        "tasks": [
            {"name": "main", "cpu_percent": 12, "stack_hwm": 512},
            {"name": "IDLE0", "cpu_percent": 80, "stack_hwm": 256},
        ],
        "sensors": {
            "dht22": {"temp_c": 23.4, "humidity_pct": 55},
        },
    }
    nodes, rels = extract_candidates(sample)
    for n in nodes:
        print(f"{n.label} {n.key} props={n.properties}")
    print("---")
    for r in rels:
        print(f"({r.source.label} {r.source.key}) -[:{r.type}]-> ({r.target.label} {r.target.key})")
