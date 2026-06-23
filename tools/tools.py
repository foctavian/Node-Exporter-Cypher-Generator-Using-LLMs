from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Literal
from ingest.mqtt_client import request_telemetry, discover, request_all_telemetry

@tool
def discover_boards() -> str:
    """
        Query the MQTT topic for all the existing boards listening on it
    """
    discovered = discover()
    if not discovered:
        return "No boards responded to discovery."
    return f"Discovered boards: {discovered}"

@tool
def query_all_boards_telemetry() -> str:
    """
        Discover all boards on the MQTT topic and retrieve telemetry for every
        one of them. Use this when asked about all boards: it finds the live
        board ids and fetches their telemetry without needing any id as input.
    """
    results = request_all_telemetry()
    if not results:
        return "No boards responded to discovery."
    lines = []
    for board_id, payload in results.items():
        lines.append(f"Board {board_id}: {payload if payload is not None else 'no telemetry (timeout)'}")
    return "\n".join(lines)

@tool
def query_espboard_telemetry(boardId: int) -> str:
    """
        Query the MQTT topic for the provided board ID to retrieve
        its telemetry data
    """
    result = request_telemetry(boardId, timeout=5.0)
    if result is None:
        return f"No telemetry from board {boardId} (timeout)"
    return result