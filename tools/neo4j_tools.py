from graph.neo4j_client import get_driver
from langchain.tools import tool


def board_exists(board_id: int) -> bool:
    driver = get_driver()
    query = "MATCH (n:BOARD) WHERE n.board_id = $boardId RETURN COUNT(n) > 0 AS exists"
    records, _, _ = driver.execute_query(query, boardId=board_id, database_="espmesh")
    return records[0]["exists"]


@tool
def check_if_entity_exists(boardId: int) -> bool:
    return board_exists(boardId)