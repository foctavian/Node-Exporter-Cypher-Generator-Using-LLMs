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

logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs.log', level=logging.INFO)

load_dotenv()
tokenizer = MistralTokenizer.v3()
mistral_model = "codestral-latest"
llm = ChatMistralAI(model = mistral_model, temperature = 0, api_key = os.getenv('MISTRAL_API_KEY'))
neo4j_uri = os.getenv('NEO4J_URI')
neo4j_username = 'neo4j'
neo4j_pass = os.getenv('NEO4J_PASS')
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_pass))
graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_pass, enhanced_schema=True)
processed_components = None
class cypher(BaseModel):
  problem: str = Field(description="problem to solve")
  cypher_script: str = Field(description="cypher script to run")

@dataclass
class SystemCypherScript(TypedDict): # this class is used to store the cypher scripts generated by the agent
  nodes: str=""
  relationships: str=""
  properties: str=""

class GraphState(TypedDict):
  error : str
  messages : List
  generation: str
  iterations : int
  script: SystemCypherScript
  traceback: bool = False
  
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

def parse_file(file):
  global components
  global processed_components
  metric_pattern = re.compile(r'(\w+)\{([^}]*)\}\s+([\d\.]+)')
  data = None
  try:
      data=clean_exporter_file(file).split('\n')
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

cypher_gen_chain = cypher_gen_prompt | structured_llm | parse_output

rel_gen_chain = relationship_gen_prompt | structured_llm | parse_output

metrics_gen_chain = metrics_gen_prompt | structured_llm | parse_output

reflections_chain = reflections_prompts | structured_llm | parse_output

node_inferring_chain = node_inferring_prompts | llm 

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
    chunk_size=10000,
    chunk_overlap=200,
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

def chunk_file(file):
  from langchain_text_splitters import CharacterTextSplitter
  with open(file, 'r') as f:
    lines = f.read() 
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=10000, chunk_overlap=200)
    texts = text_splitter.split_text(lines)
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
  
  error = 'no'
  #first validate the node script
  valid_node, msg_node = cypher_validation(nodes)
  if not valid_node: # node script is invalid
    logger.warning(f'----NODE SCRIPT VALIDATION: FAILED : {msg_node}----')
    error_message = [("user", f"Node script is invalid: {msg_node}")]
    messages += error_message
    return {'error':f'yes:nodes:{msg_node}', 'messages':messages, 'iterations':state['iterations'], 'script':solution, 'traceback':True}
  else:
    logger.info(f'----NODE SCRIPT VALIDATION: SUCCESS----')

  # then validate the relationships script
  valid_rel, msg_rel = cypher_validation(relationships)
  if not valid_rel: # relationship script is invalid
    logger.warning(f'----RELATIONSHIP SCRIPT VALIDATION: FAILED : {msg_rel}----')
    error_message = [("user", f"Relationship script is invalid: {msg_rel}")]
    messages += error_message
    return {'error':f'yes:relationships:{msg_rel}', 'messages':messages, 'iterations':state['iterations'], 'script':solution, 'traceback':True}
  else:
    logger.info(f'----RELATIONSHIP SCRIPT VALIDATION: SUCCESS----')
    
  # then validate the properties script
  property_list_query = split_query_on_semicolon(properties) # the query needs to be split to be checked individually
  wrong_queries = []
  # for q  in property_list_query[:-1]:  
  #   valid_prop, msg_prop = cypher_validation(property_list_query)
  #   if not valid_prop:
      #wrong_queries.append(q)
      #logger.warning(f'----PROPERTIES SCRIPT VALIDATION: FAILED----')
      #error_message = [("user", f"Properties script is invalid: {msg_prop}")]
      #messages += error_message
      #return {'error':f'yes:properties:{msg_prop}', 'messages':messages, 'iterations':state['iterations'], 'script':solution, 'traceback':True}
  
  logger.info(f'----PROPERTIES SCRIPT VALIDATION: SUCCESS----')
  logger.info('----CYPHER SCRIPTS VALIDATION: SUCCESS----')
  logger.info('----RUNNING SCRIPT----')
  run_cypher_query(nodes)
  run_cypher_query(relationships)
  for q in property_list_query:
    try:
      if q == '': continue
      run_cypher_query(q)
    except Exception:
      wrong_queries.append(q)
  logger.warning(f"---WRONG QUERIES: {wrong_queries}")    
  init_graph = True
  if len(wrong_queries) == 0:
    return {'error':'no', 'messages':messages, 'iterations':state['iterations'], 'script':solution, 'traceback':False}
  else:
    return {'error':f'yes:properties:{wrong_queries}', 'messages':messages, 'iterations':state['iterations'], 'script':solution, 'traceback':True}

def traceback_from_reflect(state: GraphState):
  error = state['error']
  # if 'yes' in error:
  #   _, err_cause, _ = error.split(':', 2)
  #   return err_cause 
  return 'check'
  
def existing_node(state: GraphState):
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
  logger.info('----GENERATING COMPONENTS----')
  messages = state['messages']
  iterations = state['iterations']
  error = state['error']
  script = state['script']
  traceback = state['traceback']
  
  if error == 'yes':
    messages += [
        (
                "user",
                "Now, try again. Invoke the code tool to structure the output with a problem and a cypher block:",
            )
    ]

  solution = cypher_gen_chain.invoke({
      "messages":messages
  })
  messages += [
      ('assistant', f"CYPHER: {solution}")
  ]
  
  script['nodes'] = solution.cypher_script
  logger.info(f'----GENERATED NODES: {solution.cypher_script}----')
  return {'generation':solution, "messages" : messages, "iterations" : iterations, "script": script, 'traceback':traceback}

def infer_missing_nodes(state: GraphState):
  logger.info('----INFERRING MISSING NODES----')
  
  logger.info(f'{graph.schema}') # based on this schema we can infer the missing nodes
  
  messages = state['messages']
  iterations = state['iterations']
  error = state['error']
  script = state['script']
  
  # chunk the file
   

def generate_update_script(state: GraphState):
  pass

def update_node(state:GraphState): #TODO refactor this => split into multiple nodes : infer, generate
  logger.info('----UPDATING NODES----')
  messages = state['messages']
  iterations = state['iterations']
  error = state['error']
  
  update_statement=[]
  system_name = processed_components['NODENAME'][0]
  for label in node_labels[:-1]:
    if label in metrics_component:
      metrics =' '.join(metrics_component[label])
      update_message= [
          (
              "user",
              f'''
               Task:
                  Generate Cypher queries to update existing nodes with the correct properties based on the provided metrics. If any nodes or relationships are missing, infer them based on the given data and create them before updating properties.

                  Instructions:
                    Match the system node using its name and store it with a WITH statement.
                    Match all related nodes belonging to that system using the system property.
                    Identify nodes uniquely by using both the system name and either id or name. Do not use additional properties for matching.
                    Update node properties with the exact values from the provided metrics.
                    Infer missing nodes:
                    If a referenced node does not exist, create it using the available data.
                    Ensure each inferred node has the correct system association.
                    Infer missing relationships:
                    If a relationship should exist but is missing, create it before updating properties.
                    Do not include any RETURN statements in the final queries.
                    Unpack variables properly: Always reference the system using n.name in subsequent queries.
                    Use WITH and UNWIND to carry forward variables and process multiple updates efficiently.
                    The variable names should be a combination of the label and the id or name of the node, separated by an underscore (e.g., cpu_0).

                  Property Handling Rules:
                    Extract all key-value pairs inside {{ }} as separate properties.
                    Never use maps or JSON-like structures in Cypher queries.
                    Use the exact metric values without rounding or approximating.
                    Retain values exactly as provided, including scientific notation (e.g., 3.1e+09).

                  Expected Query Structure:
                  MATCH (n:NODENAME) WHERE n.name = '{system_name}'  
                  WITH n  

                  // Match or create component node
                  MERGE (c:{{component_label}} {{system: n.name, id: '{{component_id}}'}})  
                  ON CREATE SET c.name = '{{component_name}}'  // Infer missing name if applicable  

                  // Update properties
                  SET c.PROPERTY = VALUE  
                  Use MERGE instead of MATCH to infer and create missing nodes.
                  Ensure relationships exist before updating properties.

                  Contextual Information:
                  System to be updated: {system_name}

                  Current Graph Schema:
                  {graph.schema}
                  Metrics for processing:
                  {metrics}
              '''
              )
      ]
      update = cypher_gen_chain.invoke({'messages':update_message})
  
      logger.info(f'----UPDATED NODES: {label}----')
      update_statement.append(update.cypher_script)
      logger.info(f'----UPDATE STATEMENT {label}: {update.cypher_script}----')
      if cypher_validation(update.cypher_script): # if the cypher script is valid
        logger.info('----UPDATE VALIDATION: SUCCESS----')
        run_cypher_query(update.cypher_script)
        logger.info('----UPDATE NODES: SUCCESS----')
      else: #if the cypher script is invalid, then add the error message
        logger.info('----UPDATE VALIDATION: FAILED----')
        # possible to reflect on the error 
        
      messages += [
                (
                    "assistant",
                    f"CYPHER RELATIONSHIPS: \n Problem: {update.problem} \n CYPHER: \n {update.cypher_script}",
                )
        ]
  # if cypher_validation(' '.join(update_statement)):
  #   logger.info('----UPDATE VALIDATION: SUCCESS----')
  #   run_cypher_query(update_statement)
  # else:
  #   logger.info('----UPDATE VALIDATION: FAILED----')
  return {'generation':update,"messages" : messages, "iterations" : iterations}
  

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
      if len(metrics) > 20000:
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
    #workflow.add_node('update_node', update_node)
    workflow.add_node('infer', infer_missing_nodes)
    workflow.add_node('update', generate_update_script) 
    workflow.add_node('reflect', reflect)
    
    workflow.add_edge(START, 'node_existance')
    workflow.add_conditional_edges('node_existance', check_existing_node, {True: 'infer', False: 'node_gen'}) # checks if the node already exists
    workflow.add_edge('metrics_gen', 'cypher_check')
    
    workflow.add_edge('infer', 'update')
    workflow.add_edge('update', END)
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

def start_agent(filename='node_exporter_metrics.txt'):
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
  solution = app.invoke({"messages":messages, "iterations":0, "error":"", 'script':empty_script, 'traceback':False})
  # components={label: set() for label in node_labels[:-1]}
  # metrics_component= {label: [] for label in node_labels}
  return solution

def query_graph(user_nl_query):
  print(graph.schema)
  return neo4j_chain.invoke({'query':user_nl_query})
  

#TODO implement the update node and relationship system
#TODO move all global variables to a main function and start the execution from there
#TODO change the update-node => split into two nodes, one to infer new components and  one that updates the properties of the existing nodes
#TODO move all the prompt blocks to another file 