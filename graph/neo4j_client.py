import os
import threading
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv

_driver: Driver | None = None
_lock = threading.Lock()


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        with _lock:
            if _driver is None:
                load_dotenv()
                uri = os.environ["NEO4J_URI"]
                user = os.getenv("NEO4J_USERNAME", "neo4j")
                password = os.environ["NEO4J_PASS"]
                _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def close_driver() -> None:
    global _driver
    with _lock:
        if _driver is not None:
            _driver.close()
            _driver = None
