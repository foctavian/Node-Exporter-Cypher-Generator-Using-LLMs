from collections import defaultdict
import os
import shutil
from neo4j import GraphDatabase
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph, START
from langgraph.types import Command
from typing import Literal
import re 
from typing import List, Optional
from typing_extensions import TypedDict, NotRequired
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
import logging
from PIL import Image
import tiktoken
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from dataclasses import dataclass
import httpx
from langchain_core.tools import tool
from tools import check_if_entity_exists
from models import *

logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs.log', level=logging.INFO)

############################## GLOBAL VARIABLES ############################
load_dotenv()
tokenizer = MistralTokenizer.v3()
mistral_model = "codestral-latest"
llm = ChatMistralAI(model = mistral_model,
                    temperature = 0,
                    max_tokens=20000,
                    api_key = os.getenv('MISTRAL_API_KEY'))

neo4j_uri = os.getenv('NEO4J_URI')
neo4j_username = 'neo4j'
neo4j_pass = os.getenv('NEO4J_PASS')
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_pass))
graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_pass)
processed_components = None
missing_nodes_dict = defaultdict(list)
missing_rels_dict = []
missing_props_dict=[]
  
def parse_output(solution):
  return solution["parsed"]

def run_cypher_query(query):
  from neo4j.exceptions import CypherSyntaxError
  if not query.endswith(';'):
    query+=';'
  # running the cypher query as it is, might result in an error because the driver expects
  # a single query to be executed at a time. So we split the query on the semicolon and run
  # each query individually

  try:
    driver.execute_query(query)
  except CypherSyntaxError as e:
    for q in query.split(';'):
      driver.execute_query(q)
    logger.warning(f'----CYPHER SYNTAX ERROR: {str(e)}----')

def clean_exporter_file(file):
  new_lines = []
  with open(file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('#'):
            continue
        new_lines.append(line)
  return '\n'.join(new_lines)

node_labels = ["CPU", 'GPU', 'SENSOR', 'NODENAME', 'FILESYSTEM','NETWORK','POWER_SUPPLY','DISK', 'PROPERTY']
metrics_component = {label: [] for label in node_labels}
components = {label: set() for label in node_labels[:-1]}
file_text = None

def parse_file(file):
  global components
  global processed_components
  global file_text
  
  metric_pattern = re.compile(r'(\w+)\{([^}]*)\}\s+([\d\.]+)')
  data = None
  try:
      data=clean_exporter_file(file).split('\n')
      file_text = clean_exporter_file(file) #TODO move this to another function; for now it works -> using it in the infer nodes function 
      logger.info("File parsed successfully")
  except FileNotFoundError:
      logger.error("File not found")

  for line in data:
    m = metric_pattern.match(line)
    if m:
      metric_name, properties, value = m.groups()
      props_dict = dict(re.findall(r'(\w+)="([^"]+)"', properties))
      assigned = False

      for node in node_labels[:-1]:
        if any(node.lower() in k.lower() for k,v in props_dict.items()) or node.lower() in metric_name.lower() :
          metrics_component[node].append(line)
          if node.lower() in props_dict:
            components[node].add(props_dict[node.lower()])
          elif 'device' in props_dict:
            components[node].add(props_dict['device'])

          assigned=True
          break

      if not assigned:
        metrics_component[node_labels[-1]].append(line)

  processed_components={k: list(v) for k, v in components.items()}

structured_llm = llm.with_structured_output(cypher, include_raw=True)
structured_node_inferring_llm= llm.with_structured_output(InferredNodes, include_raw=True)
structured_rel_inferring_llm = llm.with_structured_output(InferredRelationships, include_raw=True)
structured_metrics_inferring_llm = llm.with_structured_output(InferredProperties, include_raw=True)

cypher_gen_prompt = ChatPromptTemplate(
    [
        (
            "system",
            '''You are an assistant with expertise in Cypher scripts and Node exporter metrics.\n
      Ensure any code you provide can be executed with all required variables defined. \n
      Struture your answer with a description of the code solution and finally list the \n
      functioning script block.
      Here are the metrics that the user want to convert:,
            '''
        ),
        ("user", "{messages}"),
    ]
)

relationship_gen_prompt = ChatPromptTemplate(
    [
        (
            'system',
            '''
            You are an assistant with expertise in Cypher scripts and Node exporter metrics.\n
      Ensure any code you provide can be executed with all required variables defined. \n
      Struture your answer with a description of the code solution and finally list the \n
      functioning script block. You will be tasked with creating relationships between existing nodes.\n
            '''
            ),
        ('user', '{messages}')
    ]
)

metrics_gen_prompt= ChatPromptTemplate(
  [
    (
      'system',
      '''
      You are an assistant with expertise in Cypher scripts and Node exporter metrics.\n
      Ensure any code you provide can be executed with all required variables defined.\n
      Struture your answer with a description of the code solution and finally list the \n
      functioning script block. You will be tasked with creating nodes from the metrics provided.\n
      Here are the metrics that the user want to convert:
      '''
    ),
    ('user', '{messages}')
  ]
)

reflections_prompts= ChatPromptTemplate(
  [
    (
      'system',
      '''
      You are an assistant with expertise in Cypher scripts and Node exporter metrics.\n
      You will be tasked with reflecting on the error that occured during the generation of a cypher script.\n
      
      For context, the script should be part of a graph that represents a computation unit with various hardware components.\n 
      The script may be generating nodes, relationships or properties.\n
      You will be provided with the error message and the script that caused the error.\n
      '''
    ),
    ('user', '{messages}')
  ]
)

node_inferring_prompts = ChatPromptTemplate(
  [
    ('system',
     '''
      You are an assistant with expertise in Node Exporter metrics.\n
      You will be tasked with inferring the missing nodes from the metrics provided.\n

      For context, the script should be part of a graph that represents a computation unit with various hardware components.\n
     '''
     ),
    ('user', '{messages}')
  ]
)

relationship_inferring_prompts = ChatPromptTemplate(
  [
    ('system',
     '''
     You are an assistant with expertise in Node Exporter metrics.\n
      You will be tasked with inferring the relationships between from the metrics provided.\n
      
      For context, the script should be part of a graph that represents a computation unit with various hardware components.\n
     '''
     ),
    ('user', '{messages}')
  ]
)

metrics_inferring_prompts = ChatPromptTemplate(
 [
   ('system',
    '''
    You are an assistant with expertise in Node Exporter metrics.\n
    You will be tasked with inferring the missing properties from the metrics provided.\n
    For context, the script should be part of a graph that represents a computation unit with various hardware and logical components.\n
    '''
    ),
   ('user', '{messages}')
 ] 
)

cypher_gen_chain = cypher_gen_prompt | structured_llm | parse_output

rel_gen_chain = relationship_gen_prompt | structured_llm | parse_output

metrics_gen_chain = metrics_gen_prompt | structured_llm | parse_output

reflections_chain = reflections_prompts | structured_llm | parse_output

node_inferring_chain = node_inferring_prompts | structured_node_inferring_llm | parse_output 

relationships_inferring_chain = relationship_inferring_prompts | structured_rel_inferring_llm | parse_output

property_inferring_chain = metrics_inferring_prompts | structured_metrics_inferring_llm | parse_output

#tried to use this prebuilt chain but it failed to return a valid response
#for not i sticked to using a simple manual query on the db
#TODO: REVISE THIS LATER FOR FUTURE IMPROVEMENTS
neo4j_chain = GraphCypherQAChain.from_llm(
  llm,
  graph=graph,
  verbose=True,
  allow_dangerous_requests=True,
  function_response_system='Respond in a structured format with the question and the cypher query.',
  validate_cypher=True) 

#################################### AGENT UTILS ########################################

def token_threshold_check(msg:str):
    enc = tiktoken.encoding_for_model(mistral_model)
    return len(enc.encode(msg)) > 8092

def chunk_metrics(metrics:str):
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
  )
  chunks = text_splitter.split_text(metrics)
  return chunks

def cypher_validation(script):
  try:
    with driver.session() as session:
      with session.begin_transaction() as tx:
        tx.run(script)
        tx.rollback()
        return True, 'Valid'
  except Exception as e:
    return False, str(e)
  
def split_query_on_semicolon(query:str):
  return query.split(';')

def chunk_file(text):
  from langchain_text_splitters import CharacterTextSplitter 
  text_splitter = CharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=10000, chunk_overlap=200)
  texts = text_splitter.split_text(text)
  print(texts[0])
  

##################################### AGENT WORKFLOW ########################################
max_iterations = 5
flag = 'reflect'
init_graph = False
init_relationships=False
already_existing_node = False
nodes_cypher = ''

def cypher_check(state: GraphState):
  global init_graph
  logger.info('----CHECKING CYPHER----')
  messages = state['messages']
  solution = state['script']
  nodes = solution['nodes']
  relationships = solution['relationships']
  properties = solution['properties']
  if nodes:
  #first validate the node script
    valid_node, msg_node = cypher_validation(nodes)
    if not valid_node: # node script is invalid
      logger.warning(f'----NODE SCRIPT VALIDATION: FAILED : {msg_node}----')
      error_message = [("user", f"Node script is invalid: {msg_node}")]
      messages += error_message
      return {'error':f'yes:nodes:{msg_node}', 'messages':messages, 'iterations':state['iterations'], 'script':solution, 'traceback':True}
    else:
      logger.info(f'----NODE SCRIPT VALIDATION: SUCCESS----')

  if relationships:
    # then validate the relationships script
    valid_rel, msg_rel = cypher_validation(relationships)
    if not valid_rel: # relationship script is invalid
      logger.warning(f'----RELATIONSHIP SCRIPT VALIDATION: FAILED : {msg_rel}----')
      error_message = [("user", f"Relationship script is invalid: {msg_rel}")]
      messages += error_message
      return {'error':f'yes:relationships:{msg_rel}', 'messages':messages, 'iterations':state['iterations'], 'script':solution, 'traceback':True}
    else:
      logger.info(f'----RELATIONSHIP SCRIPT VALIDATION: SUCCESS----')
  
  if properties:
    # then validate the properties script
    property_list_query = split_query_on_semicolon(properties) # the query needs to be split to be checked individually
    wrong_queries = []
 
  logger.info(f'----PROPERTIES SCRIPT VALIDATION: SUCCESS----')
  logger.info('----CYPHER SCRIPTS VALIDATION: SUCCESS----')
  logger.info('----RUNNING SCRIPT----')
  if nodes:
    run_cypher_query(nodes)
  if relationships:
    run_cypher_query(relationships)
  if properties:
    for q in property_list_query:
      try:
        if q == '': continue
        run_cypher_query(q)
      except Exception:
        wrong_queries.append(q)
    logger.warning(f"---WRONG QUERIES: {wrong_queries}")
    if len(wrong_queries)>0:
      return {'error':f'yes:properties:{wrong_queries}', 'messages':messages, 'iterations':state['iterations'], 'script':solution, 'traceback':True}
          
  init_graph = True
  return {'error':'no', 'messages':messages, 'iterations':state['iterations'], 'script':solution, 'traceback':False}

def traceback_from_reflect(state: GraphState):
  error = state['error']
  # if 'yes' in error:
  #   _, err_cause, _ = error.split(':', 2)
  #   return err_cause 
  return 'check'
  
async def existing_node(state: GraphState):
    global already_existing_node
    logger.info('----CHECKING EXISTING NODE----')
    id = processed_components['NODENAME'][0]
    query=f'''MATCH (N:NODENAME) WHERE N.name='{id}' or N.id='{id}' RETURN COUNT(N)>0 as exists'''
    records, _, _=driver.execute_query(query)
    for record in records:
      if record['exists']:
        already_existing_node = True
      else:
        already_existing_node=False
    logger.info(f'RECORDS: {records}')
    logger.warning(f'query: {query}')
    logger.warning(f'Node already exists: {already_existing_node}')

def node_gen(state: GraphState) -> GraphState:
  global processed_components
  logger.info('----GENERATING COMPONENTS----')
  messages = state['messages']
  iterations = state['iterations']
  error = state['error']
  script = state['script']
  traceback = state['traceback']
  
  question = ChatPromptTemplate([
    'user',
    f'''
     Convert these components into CYPHER nodes. Make it so each label is the key of the dictionary \n
      and the name of the variable is the label followed by `_` and its id or name. For example: `cpu_0`. \n

      Do not use special characters such as `-` in variable names.
      
      For context, the  graph represents a computation unit with various hardware components.
      Because of that, I need you to add a property to each node that represents the system it is part of. For example, any node should have a property `system:"edge2-System-Product-Name"` where the value is the name of the system, represented by the node with label `NODENAME`.
      For the `NODENAME` node, do not insert a system property.
      Do not return any values in the final query.
      For every node include their name as a property. For example, the `CPU` node with id `0` should have a property `name: "0"`. 
      
      {processed_components}
    '''
  ])
  
  solution = cypher_gen_chain.invoke({
      "messages":question
  })
  
  script['nodes'] = solution.cypher_script
  logger.info(f'----GENERATED NODES: {solution.cypher_script}----')
  return {'generation':solution, "messages" : messages, "iterations" : iterations, "script": script, 'traceback':traceback}

async def infer_missing_nodes(state: GraphState):
  import time
  import asyncio
  global file_text
  logger.info('----INFERRING MISSING NODES USING ASYNC----')
  
  #logger.info(f'{graph.schema}') # based on this schema we can infer the missing nodes
  
  # chunk the file
  chunks =  chunk_metrics(file_text)
  all_messages = []
  all_responses = []
  # TEST ASYNC
  for chunk in chunks :
    prompt = f'''
                Extract only the name and type of nodes from the metrics. Use the provided schema to determine existing nodes.
                If you encounter a node that is not present in the schema, infer its name and type.
                As a rule of thumb, if the type represents a physical components, you can take it into consideration.
                Do not abbreviate the name of the label.
                Return **only** the nodes that are **not present** in the schema.
                There could be some false positives, so be careful when inferring the nodes. The schema has CPU nodes so a PROCESSOR node is not needed.
                Before you add the node, check if it is already present in the graph.
                
                Instructions:
                1. Parse `{chunk}` to extract node names and types.
                2. The label should be all CAPS and should not contain any special characters.
                3. Ignore any metrics related to the following:
                  - textfile
                  - go_*
                  - promhttp
                  - scrape_*
                  - process_*
                4. Return a **list of missing nodes**. If all nodes exist, return an **empty list**.
                5. Do not return existing nodes.
                6. There is only one NODENAME node: {processed_components['NODENAME'][0]}.
                7. Exclude providing any explanations or comments in the output.
                8. For reference, this is the graph schema: {graph.schema}
                Ensure that nodes are compared strictly by both **name** and **type** before determining if they are missing.
            '''
    all_messages.append(prompt)
    
  concurrent_batch_size = 3
  start_time = time.time()
  
  for i in range(0, len(all_messages), concurrent_batch_size):
    batch = all_messages[i:i+concurrent_batch_size]
    try:
      batch_responses = await node_inferring_chain.abatch(batch)
      all_responses.extend(batch_responses)
      if i+concurrent_batch_size < len(all_messages):
        await asyncio.sleep(1)
    except Exception as e:
      logger.error(f"Error processing batch {i//concurrent_batch_size}: {e}")
      await asyncio.sleep(5)
      for msg in batch:
        try:
          response = await node_inferring_chain.ainvoke(msg)
          all_responses.append(response)
          await asyncio.sleep(0.5)
        except Exception as inner_e:
          logger.error(f"Failed individual request: {inner_e}")
  
  end_time = time.time()      
  logger.warning(f"Batch processing took {end_time - start_time:.2f} seconds.")
  logger.warning(f"{all_responses}")

  for response in all_responses:
    for node in response.nodes:
      label = node.label
      if label not in missing_nodes_dict:
        missing_nodes_dict[node.label] = [node]
      else:
        if node not in missing_nodes_dict[label]:
          missing_nodes_dict[label].append(node)
  logger.info(f'----INFERRED NODES: {missing_nodes_dict}----')
  return {**state}

#TODO add a check to see if the node or relationship is already present in the graph  

def infer_missing_relationships(state: GraphState):
  import time
  logger.info('----INFERRING MISSING RELATIONSHIPS----')
  global file_text
  global missing_rels_dict
  missing_relationships = []
  chunks = chunk_metrics(file_text)
  time.sleep(1)
  # use the chunk to infer the missing nodes
  messages = [
      (
          "user",
          f'''
          Infer the relationships between the nodes based on the provided metrics. Use the provided schema to determine existing relationships.
          These are the nodes that were previously inferred: 
          {state['script']['nodes']}
          This is the schema of the graph:
          {graph.schema}
          
          Steps:
          1. Parse the created nodes to extract the missing relationships.
          2. Compare these against the provided schema (`{graph.schema}`).
          3. Return a **list of missing relationships**. If all relationships exist, return an **empty list**.
          4. Do not return existing relationships.
          5. The relationship should be in the format: `HAS_*`. 
          6. The only node that should not have a system property is the `NODENAME` node. Insert an empty string for the system property.
          
          '''
      )
  ]
  text = "".join([content for _, content in messages])
  logger.info(f'----Message size: {len(text)}----')
    
  solution = relationships_inferring_chain.invoke({
        "messages":messages
    })
  if not solution:
    logger.info('----NO MISSING RELATIONSHIPS----')
    messages += [
        ('assistant', 'NO MISSING RELATIONSHIPS')
    ]
  else:
    logger.info(f'----INFERRED RELATIONSHIPS: {solution.relationships}----')
    messages += [
        ('assistant', f"INFERRED RELATIONSHIPS: {solution.relationships}")
    ]
    missing_relationships += solution.relationships
  missing_rels_dict=missing_relationships
  return {**state}
  

def generate_node_update_script(state:GraphState): # TODO route this to the existing node generation node
  question = ChatPromptTemplate([
      ('user',
      f"""
    Convert these components into Cypher statements to create nodes using `MERGE` only. 
    Do not use `CREATE`.

    For **each** node, use the following pattern:
    ```
    MERGE (var:Label {{system: "system_name", name: "node_name"}})
    ON CREATE SET var.createdAt = timestamp()
    ON MATCH SET var.lastSeen = timestamp()
    ```

    - Variable names should follow the format: `label_idOrName` (e.g., `cpu_0`)
    - Do not use special characters like `-` in variable names
    - Add a property `system: "{processed_components['NODENAME'][0]}"` to every node **except** the `NODENAME` node
    - Always include `name: "..."` as a property on the node
    - Do **not** return any output values in the final query

    For context, the graph represents a computation unit with various hardware components.

    Input:
    {missing_nodes_dict}
    """
      )
  ]
  )
  
  solution = cypher_gen_chain.invoke({
      "messages":question
  })
  
  #solution.cypher_script = solution.cypher_script.replace('CREATE ', 'MERGE ')
  
  logger.info(f'----GENERATED UPDATED NODES: {solution.cypher_script}----')
  state['script']['nodes'] = solution.cypher_script
  
  return {'script':state['script'], **state}

def infer_missing_metrics(state: GraphState):
  import time
  logger.info('----INFERRING MISSING METRICS----')
  global file_text
  global missing_props_dict
  missing_props = []
  chunks = chunk_metrics(file_text)
  
  for chunk in chunks :
    time.sleep(0.5)
    
    messages = [
      (
        "user",
        f'''
        Infer the properties of the nodes based on the provided metrics. Use the provided schema to determine existing properties and nodes.
        This is the schema of the graph:
        {graph.schema}
        These are the nodes that were previously inferred:
        {state['script']['nodes']}
        
        Steps:
        1. Parse `{chunk}` to extract property names and values.
        2. Return a **list of missing properties**. If all properties exist, return an **empty list**.
        3. Do not return existing properties.
        
        '''
      )
    ]
    
    solution = property_inferring_chain.invoke({
        "messages":messages
    })
    
    if not solution:
      logger.info('----NO MISSING PROPERTIES----')
    else:
      logger.info(f'----INFERRED PROPERTIES: {solution.properties}----')
      messages += [
          ('assistant', f"INFERRED PROPERTIES: {solution.properties}")
      ]
      missing_props_dict += solution.properties
  
  return {**state}
    

def generate_metric_update_script(state: GraphState):
  logger.info(f'GENERATING METRICS UPDATE SCRIPT : {len(missing_props_dict)}')
  script = state['script']
  script['properties'] = ''
  def chunk_generator(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]
  
  chunk_size = 100
  
  for chunk in chunk_generator(missing_props_dict, chunk_size):
    question = ChatPromptTemplate([
        ('user',
        f"""
        Convert the following properties into Cypher `MERGE`-based update statements for existing nodes.  
        Each node is uniquely identified by its `label`, `name`, and `system`.

        Instructions:
        1. Use `MATCH` to locate the node by its `label`, filtering by both `name` and `system` properties.
        2. Use `SET` to update the specified properties on the matched node.
        3. Include `n.updatedAt = timestamp()` in every `SET` clause.
        4. Output **only** the final Cypher code — do not include comments or explanations.
        5. Do **not** include any `RETURN` statements.
        6. Follow this exact pattern for each node:

        MATCH (n:Label {{name: "node_name", system: "system_name"}})
        SET n.property = value, n.updatedAt = timestamp()


        Context:
        The graph models a computation unit composed of multiple components.

        Input:
        {chunk}
      """
        )
    ]
    )
    solution = metrics_gen_chain.invoke({
        "messages":question
    })  
    script['properties'] += solution.cypher_script
  logger.info(f'----GENERATED UPDATED METRICS: {script['properties']}----')
  return {'script':script, **state}

def generate_rel_update_script(state: GraphState):
  question = ChatPromptTemplate([
      ('user',
      f"""
      Convert the given relationship objects into Cypher statements that define directed relationships between existing nodes in a Neo4j graph.
      Do not use `CREATE` — only use `MATCH` and `MERGE`.

      Instructions:

      1. Use `MATCH` to locate both source and target nodes.
      2. Use the labels of the nodes to identify them and filter by their `name` and `system` properties. The only node that should not have a system property is the `NODENAME` node.
      3. Match pattern:
         `MATCH (n:label {{name: "node_name", system: "system_name"}})`
      4. Use `MERGE` to define a directed relationship from the source node to the target node.
      5. The relationship type must be the **uppercase** version of the `relationship` field.
      6. Ensure each relationship is explicitly **directed from source to target**.
      7. If any node appears disconnected or isolated, infer a logical link to existing nodes based on its type (e.g., cooling device → CPU or SYSTEM).
      8. Do not include any `RETURN` statements or explanatory text — only output the Cypher code.

      Context:
      
      The graph represents a computing unit composed of interconnected hardware components (e.g., CPU, memory, GPU, sensors, etc.).

      Input:
      {missing_rels_dict}
    """
      )
  ]
  )
  
  solution = rel_gen_chain.invoke({
      "messages":question
  })
  
  logger.info(f'----GENERATED UPDATED RELATIONSHIPS: {solution.cypher_script}----')
  script = state['script']
  script['relationships'] = solution.cypher_script
  return {'script':script, **state}

def component_validation(state: GraphState):
  print('----VALIDATING COMPONENTS----')
  messages = state['messages']
  iterations = state['iterations']
  error = state['error']
  solution = state['generation']

  valid, msg = cypher_validation(solution.cypher_script)
  if valid == False:
    logger.info('----NODE GENERATION VALIDATION: FAILED----')
  else:
    logger.info('----NODE GENERATION VALIDATION: SUCCESS----')

def run_script(state: GraphState):
  from neo4j.exceptions import CypherSyntaxError
  logger.info('----RUNNING SCRIPT----')
  #print(state['generation'])
  node_gen_script = state['generation'].cypher_script
  cleaned_script = " ".join(node_gen_script.split())
  if not cleaned_script.endswith(';'):
    cleaned_script+=';'

  cleaned_script.replace('", "', "")
  
  try:
    driver.execute_query(cleaned_script)
  except CypherSyntaxError as e:
    for query in cleaned_script.split(';')[:-1]:
      driver.execute_query(query)
      
def relationships_gen(state: GraphState):
  global nodes_cypher
  logger.info('----GENERATING RELATIONSHIPS----')

  #Test
  # as a first test i would like to generate the relationships
  # between the nodename and the rest of the components
  messages = state['messages']
  iterations = state['iterations']
  error = state['error']
  script = state['script']
  traceback = state['traceback']
  nodes = state['generation'].cypher_script
  nodes_cypher= ''.join(nodes)
  rel_message= [
      (
          "user",
          f'''
            Generate CYPHER relationships between the nodes that were previously created.
            Create meaningful relationships. For context, the graph that I want to create represents
            a node in a computation unit having different components.
            The relationship names should not be ambiguous. If it is only composed of `has` concatenate the
            name of the label using `_`.

            Only create relationships between existing nodes, do not create new nodes.
            First match all the nodes that are part of the system and then create the relationships. Carry on the variables using WITH and UNWIND.
            Use this syntax to create relationships:
            `MATCH (n:{{node_label}})
            MATCH (c:{{component_1_label}} {{system: '{{system_identifier}}'}})
            MATCH (s:{{component_2_label}} {{system: '{{system_identifier}}'}})
            MATCH (d:{{component_3_label}} {{system: '{{system_identifier}}'}})
            WITH n, COLLECT(c) AS cpus, COLLECT(s) AS sensors, COLLECT(d) AS disks
            FOREACH (cpu IN cpus | MERGE (n)-[:{{relationship_1}}]->(cpu))
            FOREACH (sensor IN sensors | MERGE (n)-[:{{relationship_2}}]->(sensor))
            FOREACH (disk IN disks | MERGE (n)-[:{{relationship_3}}]->(disk));
            `
            
            For context, the generated graph describes a computation unit that has different components.
            The relationships should start from the main node and have the following format: `HAS_*`.
            The nodes that are part of a system should have a property `system` that represents the system it is part of. Use that to choose the nodes to connect.
            For example, if the node with label `CPU` has a property `system: "edge2-System-Product-Name"`, then connect it to the NODENAME with that name.
            The NODENAME node should not have a system property, but a name property that represents the name of the system. Use that to create the relationships.
            
            These are the previously generated nodes:
            {nodes}
          '''
          )
  ]
  relationships = rel_gen_chain.invoke({'messages':rel_message})

  messages += [
        (
            "assistant",
            f"CYPHER RELATIONSHIPS: \n Problem: {relationships.problem} \n CYPHER: \n {relationships.cypher_script}",
        )
    ]

  script['relationships'] = relationships.cypher_script
  logger.info(f'----GENERATED RELATIONSHIPS: {relationships.cypher_script}----')
  #print("Generated relationships:", relationships)
  return {'generation':relationships,"messages" : messages, "iterations" : iterations, "script":script, 'traceback':traceback}

def metrics_gen(state:GraphState): #TODO: refactor this code to be more modular ; also change name to be more specific
  global nodes_cypher
  global iterations
  global init_graph
  logger.info('----GENERATING METRICS----')
  generated_metrics_script=[]

  messages = state['messages']
  iterations = state['iterations']
  error = state['error']
  script = state['script']
  traceback = state['traceback']
  
  
  if 'yes' in error:
    _, _, err_msg = error.split(':', 2)
    logger.info('----METRICS RETRYING GENERATION----')
    messages += [
        (
                "user",
                f'''Now, try again. Fix this error: {err_msg}. \n
                Provide the whole query with the error fixed. \n
                ''',
            )
    ]
    
    regenerated_metrics = metrics_gen_chain.invoke({
        "messages":messages
    })
    script['properties'] = regenerated_metrics.cypher_script
    logger.info(f'----REGENERATED METRICS: {regenerated_metrics.cypher_script}----')
    messages += [
        ('assistant', f"METRICS: {regenerated_metrics}")
    ]
    init_graph=True
    return {'generation':regenerated_metrics, "messages" : messages, "iterations" : iterations, "script":script, 'traceback':traceback} 
  
  for label in node_labels[:-1]:
    if label=='GPU':
      continue
    if label in metrics_component:
      metrics =' '.join(metrics_component[label])
      logger.warning(f'METRICS: {label} SIZE {len(metrics)}')
      if len(metrics) > 5000:
        chunks = chunk_metrics(metrics)
        for chunk in chunks:
          metrics_message=[
            (
              "user",
              f'''
            Generate Cypher queries to insert metrics into an existing Neo4j graph.

            The metrics should be added as properties of the appropriate nodes that are already created in the graph.
            Use the following syntax to update node properties:
              `MATCH (n:LABEL {{id: VALUE}} {{n.system=`system_name`}}) SET n.PROPERTY = METRIC_VALUE`
            Example: Given the metric node_cpu_frequency_max_hertz{{cpu="0"}} 3.1e+09, update the node with label CPU and id: "0" by setting max_hertz = 3.1e+09.
            Each key-value pair inside {{}} must be set as an individual property. Do not treat them as a single map.
            If a metric has multiple attributes (e.g., node_network_info{{address="02:42:db:56:1c:74", adminstate="up"}}), split them into separate properties like
              `MATCH (n:Network {{n.system=`system_name`}}) SET n.address = "02:42:db:56:1c:74", n.adminstate = "up"`    
            **Rules to follow:**
            - Never use maps or JSON-like structures in Cypher queries. Each attribute must be a separate property.
            - **Do not use `WITH` or `UNWIND` statements.**
            - **Ensure every `MATCH` query ends with `;`** before generating the next one.
            - **Batch updates:** If multiple properties are set for the same node, use a **single `MATCH` statement** and multiple `SET` clauses.
            - Do not approximate values—use the exact values provided.
            - Never return anything; no `RETURN` statements.
            - Use the name or id of the NODENAME node to match the correct nodes. Every node has a property that represents the system it's part of.

            Existing Nodes:
            {nodes_cypher}

            Metrics to Convert:
            {chunk}
              '''
            )
          ]
          logger.info(f'----GENERATED METRICS: {label}----')
          gen_metrics = metrics_gen_chain.invoke({'messages':metrics_message})
          if gen_metrics:
            generated_metrics_script.append(gen_metrics.cypher_script.replace('\n', ''))
      else:
        metrics_message=[
          (
            "user",
            f'''
          Generate Cypher queries to insert metrics into an existing Neo4j graph.

          The metrics should be added as properties of the appropriate nodes that are already created in the graph.
          Use the following syntax to update node properties:
            `MATCH (n:LABEL {{id: VALUE}} {{n.system=`system_name`}}) SET n.PROPERTY = METRIC_VALUE`
          Example: Given the metric node_cpu_frequency_max_hertz{{cpu="0"}} 3.1e+09, update the node with label CPU and id: "0" by setting max_hertz = 3.1e+09.
          Each key-value pair inside {{}} must be set as an individual property. Do not treat them as a single map.
          If a metric has multiple attributes (e.g., node_network_info{{address="02:42:db:56:1c:74", adminstate="up"}}), split them into separate properties like
            `MATCH (n:Network {{n.system=`system_name`}}) SET n.address = "02:42:db:56:1c:74", n.adminstate = "up"`    
           **Rules to follow:**
            - Never use maps or JSON-like structures in Cypher queries. Each attribute must be a separate property.
            - **Do not use `WITH` or `UNWIND` statements.**
            - **Ensure every `MATCH` query ends with `;`** before generating the next one.
            - **Batch updates:** If multiple properties are set for the same node, use a **single `MATCH` statement** and multiple `SET` clauses.
            - Do not approximate values—use the exact values provided.
            - Never return anything; no `RETURN` statements.
          
          For context, the  graph represents a computation unit with various hardware components. 

          Existing Nodes:
          {nodes_cypher}

          Metrics to Convert:
          {metrics}
            '''
          )
        ]
        logger.info(f'----GENERATED METRICS: {label}----')
        gen_metrics = metrics_gen_chain.invoke({'messages':metrics_message})
        if gen_metrics:
          generated_metrics_script.append(gen_metrics.cypher_script.replace('\n', ''))
    
  messages += [
        (
            "assistant",
            f"CYPHER METRICS: \n {' '.join(generated_metrics_script)}",
        )
    ]
  
  script['properties'] = " ".join(generated_metrics_script)
    
  init_graph=True
  solution=cypher(problem='Metrics', cypher_script=' '.join(generated_metrics_script))
  return {'generation':solution, "messages" : messages, "iterations" : iterations, "script":script, 'traceback':traceback} 
  
def reflect(state: GraphState): #TODO might need to be refactored; the if statement is not needed
  messages = state['messages']
  iterations = state['iterations']
  error = state['error']
  solution = state['generation']
  traceback = state['traceback']
  script = state['script']
  logger.info(f"----ENTERED REFLECT: {'yes' in error}----")
  if 'yes' in error:
    _, err_cause, err_msg = error.split(':', 2)# too many values to unpack => the error contains some more ':'
    logger.info('----REFLECTING ON ERROR----')
    messages += [
        (
                "user",
                f"""Reflect on the error that occured during {err_cause} generation: {err_msg}\n 
                Provide the fixed cypher script with the error fixed. \n
                """,
            )
    ]
    solution = reflections_chain.invoke({
        "messages":messages
    })
    messages += [
        ('assistant', f"These are the reflections: {solution}")
    ]
    traceback=True
    logger.warning(f'----REFLECTIONS: {solution}----')
    if solution:
      script['properties'] = solution.cypher_script
    return {'error':error, 'messages':messages, 'iterations':iterations, 'generation':solution, 'script':script, 'traceback':traceback}

def check_traceback_reflect_error(state: GraphState):
  return state['traceback']

def check_end(state: GraphState):
  global init_graph, metrics_component, components
  error = state['error']
  logger.warning(f'----CHECKING END OF WORKFLOW: {error}----')
  if init_graph and 'yes' not in error:
    logger.info('----END OF WORKFLOW----')
    logger.info('----WORKFLOW SUCCESSFULLY COMPLETED----')
    # clean the components and metrics
    for label in node_labels:
      metrics_component[label].clear()
    for label in node_labels[:-1]:
      components[label].clear()  
    return 'end'
  elif 'yes' in error:
    logger.info('----WORKFLOW FAILED: REDIRECTING TO REFLECT----')
    return 'reflect'

def check_existing_node(state: GraphState):
    global already_existing_node
    return already_existing_node

def inititialize_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node('node_existance', existing_node)
    workflow.add_node('node_gen', node_gen)
    workflow.add_node('relationship_gen', relationships_gen)
    workflow.add_node('metrics_gen', metrics_gen)
    workflow.add_node('cypher_check', cypher_check)
    workflow.add_node('reflect', reflect)

    workflow.add_node('infer_node', infer_missing_nodes) # go to node_gen if there are missing nodes, else generate only the metrics
    workflow.add_node('update_node', generate_node_update_script) 
    workflow.add_node('infer_relationship', infer_missing_relationships)
    workflow.add_node('update_relationship', generate_rel_update_script)
    workflow.add_node('infer_metrics', infer_missing_metrics)
    workflow.add_node('update_metrics', generate_metric_update_script)
    
    workflow.add_edge(START, 'node_existance')
    workflow.add_conditional_edges('node_existance', check_existing_node, {True: 'infer_node', False: 'node_gen'}) # checks if the node already exists
    workflow.add_edge('metrics_gen', 'cypher_check')
    
    workflow.add_edge('infer_node', 'update_node') 
    workflow.add_edge('update_node', 'infer_relationship') 
    workflow.add_edge('infer_relationship', 'update_relationship') 
    workflow.add_edge('update_relationship', "infer_metrics") 
    workflow.add_edge('infer_metrics', 'update_metrics') 
    workflow.add_edge('update_metrics', 'cypher_check') 
    workflow.add_conditional_edges('node_gen' ,check_traceback_reflect_error, {True: 'cypher_check', False: 'relationship_gen'})
    workflow.add_conditional_edges('relationship_gen', check_traceback_reflect_error, {True: 'cypher_check', False: 'metrics_gen'})    
    
    workflow.add_conditional_edges(
      "cypher_check",
      check_end,
      {
        "end": END,
        "reflect": 'reflect',
      }
    )
    
    workflow.add_conditional_edges(
      'reflect',
      traceback_from_reflect,
      {
        'check': 'cypher_check',
        # 'node': 'node_gen',
        # 'relationship': 'relationship_gen',
        # 'properties': 'metrics_gen'
      }
    )

    app = workflow.compile()
    return app

def visualize_graph(app, output_file='graph.png'):
  from IPython.display import Image, display
  from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles  
  app.get_graph().draw_mermaid_png(
        output_file_path=output_file,  # Save the image
        draw_method=MermaidDrawMethod.API,  # Use API method
        curve_style=CurveStyle.LINEAR,
        node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
        wrap_label_n_words=9,
        background_color="white",
        padding=10,
    )

    # Check if the file was saved
  if os.path.exists(output_file):
      print(f"Graph saved as {output_file}")
  else:
      print("Graph generation failed.")

async def start_agent(filename='node_exporter_metrics.txt'):
  parse_file(filename)
  global processed_components
  global metrics_component
  app = inititialize_graph()
  #visualize_graph(app)
  logger.info('----STARTING AGENT----')
  question = ChatPromptTemplate([
      ('user',
      '''
      Convert these components into CYPHER nodes. Make it so each label is the key of the dictionary \n
      and the name of the variable is the label followed by `_` and its id or name. For example: `cpu_0`. \n

      Do not use special characters such as `-` in variable names.
      
      For context, the  graph represents a computation unit with various hardware components.
      Because of that, I need you to add a property to each node that represents the system it is part of. For example, any node should have a property `system:"edge2-System-Product-Name"` where the value is the name of the system, represented by the node with label `NODENAME`.
      For the `NODENAME` node, do not insert a system property.
      Do not return any values in the final query.
      For every node include their name as a property. For example, the `CPU` node with id `0` should have a property `name: "0"`. 
      
      {processed_components}
      '''
      )
  ]
  )
  messages = question.format_messages(processed_components=processed_components)
  if not isinstance(messages , list):
      messages = [messages]
  empty_script = SystemCypherScript()
  solution = await app.ainvoke({"messages":messages, "iterations":0, "error":"", 'script':empty_script, 'traceback':False})
  return solution

def query_graph(user_nl_query):
  print(graph.schema)
  return neo4j_chain.invoke({'query':user_nl_query})

