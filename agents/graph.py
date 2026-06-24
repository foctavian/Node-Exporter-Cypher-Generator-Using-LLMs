import json
import re
from typing import TypedDict, Annotated, LiteralString, cast
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from .fetch import get_fetch_agent
from .extract import extract_candidates, CandidateNode, CandidateRel
from graph.neo4j_client import get_driver


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    telemetry: str | None
    cypher: str | None
    nodes: list[CandidateNode] | None
    rels: list[CandidateRel] | None
    error: str | None


def ask(agent, prompt: str) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return result


# REDO IF MEMORY IS ADDED
def _telemetry_payloads(messages) -> list[dict]:
    payloads = []
    for msg in messages:
        if getattr(msg, "type", None) != "tool":
            continue
        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue
        for item in (data if isinstance(data, list) else [data]):
            if isinstance(item, dict) and "board_id" in item:
                payloads.append(item)
    return payloads


def extract_node(state: GraphState) -> dict:
    payloads = _telemetry_payloads(state["messages"])
    if not payloads:
        return {"nodes": None, "rels": None}
    nodes, rels = [], []
    for payload in payloads:
        p_nodes, p_rels = extract_candidates(payload)
        nodes.extend(p_nodes)
        rels.extend(p_rels)
    return {"nodes": nodes, "rels": rels}

_CYPHER_SYSTEM_PROMPT = """
    You generate Neo4j Cypher that builds a graph of ESP boards
    from extracted nodes and relationships.

    Rules:
    - Use MERGE on each node's key properties for identity, then SET its other properties.
    - Create relationships with MERGE using the given relationship types.
    - Do not use special characters such as `-` in variable names.
    - Do not return any values in the final query.
"""

_cypher_prompt = ChatPromptTemplate.from_messages([
    ("system", _CYPHER_SYSTEM_PROMPT),
    ("user", "Nodes:\n{nodes}\n\nRelationships:\n{rels}"),
])

_cypher_chain = None


def _get_cypher_chain():
    global _cypher_chain
    if _cypher_chain is None:
        load_dotenv()
        llm = init_chat_model("mistralai:codestral-latest", temperature=0)
        _cypher_chain = _cypher_prompt | llm
    return _cypher_chain


def graph_generation_node(state: GraphState) -> dict:
    nodes = state["nodes"] or []
    rels = state["rels"] or []
    if not nodes:
        return {"cypher": None}
    result = _get_cypher_chain().invoke({
        "nodes": json.dumps([n.model_dump() for n in nodes], indent=2),
        "rels": json.dumps([r.model_dump() for r in rels], indent=2),
    })
    return {"cypher": result.content}

def _strip_cypher_fences(text: str) -> str:
    """Pull the Cypher out of a possibly markdown-fenced LLM response."""
    fence = re.search(r"```(?:cypher)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fence.group(1) if fence else text).strip()


def cypher_runner_node(state: GraphState) -> dict:
    cypher = state.get("cypher")
    if not cypher:
        return {"error": "no cypher to run"}

    script = _strip_cypher_fences(cypher)
    statements = [s.strip() for s in script.split(";") if s.strip()]
    if not statements:
        return {"error": "no cypher statements after parsing"}

    driver = get_driver()
    print('__________STATEMENTS')
    print(statements)
    try:
        with driver.session(database="espmesh") as session:
            with session.begin_transaction() as tx:
                for stmt in statements:
                    tx.run(cast(LiteralString, stmt))
                tx.commit()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"error": None}



#TODO: add the check for existing nodes in the db -> if they exist compute the diff from the received message
def create_workflow():
    builder = StateGraph(GraphState)

    builder.add_node('fetch_mesh_data', get_fetch_agent())
    builder.add_node('extract', extract_node)
    builder.add_node('graph_generation', graph_generation_node)
    builder.add_node('cypher_runner', cypher_runner_node)
    
    builder.add_edge(START, 'fetch_mesh_data')
    builder.add_edge('fetch_mesh_data', 'extract')
    builder.add_edge('extract', 'graph_generation')
    builder.add_edge('graph_generation', 'cypher_runner' )
    builder.add_edge('cypher_runner', END)
    
    return builder.compile()


def main():
    workflow = create_workflow()
    result = ask(workflow, "Return the telemetry for all boards.")
    
    nodes = result.get("nodes")
    if not nodes:
        print("No telemetry candidates extracted.")
        return
    for n in nodes:
        print(f"{n.label} {n.key} props={n.properties}")
    for r in result.get("rels") or []:
        print(f"({r.source.label} {r.source.key}) -[:{r.type}]-> ({r.target.label} {r.target.key})")
    # print("--- cypher ---")
    # print(result.get("cypher"))
    # print("--- runner error ---")
    # print(result.get("error"))


if __name__ == '__main__':
    main()
