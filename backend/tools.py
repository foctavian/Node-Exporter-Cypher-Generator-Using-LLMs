"""
AGENT TOOLS 
"""

from langchain_core.tools import tool
from models import Node
from neo4j import GraphDatabase
import os

@tool
def check_if_entity_exists(driver, entity:Node):
  """
  Check if the entity exists in the graph.
  """
  query = f'''
    MATCH (n:{entity.label}) WHERE n.name='{entity.name}' RETURN COUNT(n)>0 as exists
  '''
  neo4j_uri = os.getenv('NEO4J_URI')
  neo4j_username = 'neo4j'
  neo4j_pass = os.getenv('NEO4J_PASS')
  driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_pass))
  records, _, _=driver.execute_query(query)
  for record in records:
    if record['exists']:
      return True
    else:
      return False
  