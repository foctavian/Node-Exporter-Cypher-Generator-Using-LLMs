import os
import ssl
import json
import time
import threading
from paho.mqtt import client as mqtt_client
from paho.mqtt.enums import CallbackAPIVersion
from dotenv import load_dotenv

load_dotenv()
BROKER = os.environ["MQTT_BROKER"]
PORT = int(os.getenv("MQTT_PORT", "8883"))
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "ingest-service")

TELEMETRY_TOPIC = "mesh/telemetry"

_client: mqtt_client.Client | None = None
_lock = threading.Lock()


_pending: dict[int, dict] = {}
_pending_lock = threading.Lock()

_discovery_active = False
_discovered: set[int] = set()
_discovery_lock = threading.Lock()
_discovered_ids = []

def get_client():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = _create_client()
    return _client

def _create_client():
    client = mqtt_client.Client(
        client_id=CLIENT_ID,
        callback_api_version=CallbackAPIVersion.VERSION2
    )

    if USERNAME:
        client.username_pw_set(USERNAME, PASSWORD)

    client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)
    client.on_message = _on_message
    client.connect(BROKER, PORT)
    client.loop_start()

    client.subscribe(TELEMETRY_TOPIC, qos=1)

    return client

def _extract_board_id(payload: str) -> int | None:
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
    board_id = data.get("board_id", data.get("boardId"))
    return int(board_id) if board_id is not None else None

def _on_message(client, userdata, msg):
    payload = msg.payload.decode()
    board_id = _extract_board_id(payload)
    if board_id is None:
        return

    with _discovery_lock:
        if _discovery_active:
            _discovered.add(board_id)

    with _pending_lock:
        record = _pending.get(board_id)
    if record is not None:
        record["payload"] = payload
        record["event"].set()

def send_cmd(boardId: int, cmd: str):
    global _client
    if _client is not None:
        _client.publish(f"mesh/cmd/{cmd}", json.dumps({"board_id": boardId}))

def request_telemetry(boardId: int, timeout: float = 5.0) -> str | None:
    client = get_client()

    event = threading.Event()
    with _pending_lock:
        _pending[boardId] = {"event": event, "payload": None}

    client.publish("mesh/cmd/telemetry", json.dumps({"board_id": boardId}))

    got = event.wait(timeout)

    with _pending_lock:
        record = _pending.pop(boardId, None)

    if got and record is not None:
        return record["payload"]
    return None

def _get_discovered_boardIds() -> list[int]:
    global _discovered_ids
    return _discovered_ids

def discover(window: float = 3.0) -> list[int]:
    global _discovery_active
    client = get_client()

    with _discovery_lock:
        _discovered.clear()
        _discovery_active = True

    client.publish("mesh/cmd/discover", "")

   
    time.sleep(window)

    with _discovery_lock:
        _discovery_active = False
        result = sorted(_discovered)
        _discovered.clear()
    
    global _discovered_ids
    _discovered_ids = result
    return result

def request_all_telemetry(window: float = 3.0, timeout: float = 5.0) -> dict[int, str | None]:
    """Discover the live boards and fetch telemetry for each, entirely in
    Python. Board ids come from discovery, never from the caller, so there are
    no hallucinated ids. Returns {boardId: payload-or-None}."""
    boards = discover(window=window)
    return {board_id: request_telemetry(board_id, timeout=timeout) for board_id in boards}
