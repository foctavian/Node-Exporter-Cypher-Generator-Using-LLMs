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
from typing_extensions import TypedDict
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
import logging
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs.log', level=logging.INFO)

load_dotenv()
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

class GraphState(TypedDict):
  error : str
  messages : List
  generation: str
  iterations : int

def parse_output(solution):
  return solution["parsed"]

def run_cypher_query(query):
  from neo4j.exceptions import CypherSyntaxError
  if not query.endswith(';'):
    query+=';'

  
  try:
    driver.execute_query(query)
  except CypherSyntaxError as e:
    logger.warning(f'----CYPHER SYNTAX ERROR: {str(e)}----')
    for query in query.split(';'):
      driver.execute_query(query)

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

cypher_gen_chain = cypher_gen_prompt | structured_llm | parse_output

rel_gen_chain = relationship_gen_prompt | structured_llm | parse_output

metrics_gen_chain = metrics_gen_prompt | structured_llm | parse_output

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


##################################### AGENT WORKFLOW ########################################
max_iterations = 5
flag = 'reflect'
init_graph = False
init_relationships=False
already_existing_node = False
nodes_cypher = ''

def cypher_validation(script):
  try:
    with driver.session() as session:
      with session.begin_transaction() as tx:
        tx.run(script)
        tx.rollback()
        return True, 'Valid'
  except Exception as e:
    return False, str(e)

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

def component_gen(state: GraphState) -> GraphState:
  print('----GENERATING COMPONENTS----')
  messages = state['messages']
  iterations = state['iterations']
  error = state['error']

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

  return {'generation':solution, "messages" : messages, "iterations" : iterations}

def update_node(state:GraphState):
  logger.info('----UPDATING NODES----')
  messages = state['messages']
  iterations = state['iterations']
  update_statement=[]
  system_name = processed_components['NODENAME'][0]
  for label in node_labels[:-1]:
    if label in metrics_component:
      metrics =' '.join(metrics_component[label])
      update_message= [
          (
              "user",
              f'''
                Generate CYPHER queries to update existing nodes with the correct properties based on the provided metrics.

                INSTRUCTIONS:
                1. First, match the system node using its name and save it with a WITH statement
                2. Then match all nodes belonging to that system using the system property
                3. Use the node's unique identifier (id or name) alongside the system property when matching specific nodes
                4. Update node properties using the exact values from the metrics
                5. If nodes or relationships are missing, create them before updating properties
                6. Do not include any RETURN statements in the final queries

                RULES FOR PROPERTY HANDLING:
                - For each metric, parse all key-value pairs inside {{}} as separate properties
                - Never use maps or JSON-like structures in Cypher queries
                - Use the exact values provided without approximation or rounding
                - For scientific notation (e.g., 3.1e+09), use the exact value as provided
                
                Use this syntax:
                `
                MATCH(n:NODENAME) WHERE n.name = '{system_name}'
                WITH n
                MATCH (c:{{component_label}} {{system: n, id: '{{component_id}}'}})
                SET c.PROPERTY = VALUE
                `
                
                First match all the nodes that are part of the system and then update their properties. Carry on the variables using WITH and UNWIND.
                The system that needs to be updated is:
                {system_name}
                This is the current graph schema:
                {graph.schema}
                
                These are the metrics that you need to process:
                {metrics}
              '''
              )
      ]
      update = cypher_gen_chain.invoke({'messages':update_message})
  
      logger.info(f'----UPDATED NODES: {update.cypher_script}----')
      update_statement.append(update.cypher_script)
      if cypher_validation(update.cypher_script):
        logger.info('----UPDATE VALIDATION: SUCCESS----')
        run_cypher_query(update.cypher_script)
      else:
        logger.info('----UPDATE VALIDATION: FAILED----')
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
      
def generate_relationships(state: GraphState):
  global nodes_cypher
  logger.info('----GENERATING RELATIONSHIPS----')

  #Test
  # as a first test i would like to generate the relationships
  # between the nodename and the rest of the components
  messages = state['messages']
  iterations = state['iterations']
  error = state['error']


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
  logger.warning(f'----GENERATED RELATIONSHIPS: {relationships.cypher_script}----')
  messages += [
        (
            "assistant",
            f"CYPHER RELATIONSHIPS: \n Problem: {relationships.problem} \n CYPHER: \n {relationships.cypher_script}",
        )
    ]

  #print("Generated relationships:", relationships)
  return {'generation':relationships,"messages" : messages, "iterations" : iterations}

def generate_metrics(state:GraphState):
  global nodes_cypher
  global iterations
  global init_graph
  logger.info('----GENERATING METRICS----')
  generated_metrics_script=[]

  messages = state['messages']
  iterations = state['iterations']
  error = state['error']
  
  for label in node_labels[:-1]:
    if label=='GPU':
      continue
    if label in metrics_component:
      metrics =' '.join(metrics_component[label])
      metrics_message=[
        (
          "user",
          f'''
        Generate Cypher queries to insert metrics into an existing Neo4j graph.

        The metrics should be added as properties of the appropriate nodes that are already created in the graph.
        Use the following syntax to update node properties:
          `MATCH (n:LABEL {{id: VALUE}}) SET n.PROPERTY = METRIC_VALUE`
        Example: Given the metric node_cpu_frequency_max_hertz{{cpu="0"}} 3.1e+09, update the node with label CPU and id: "0" by setting max_hertz = 3.1e+09.
        Each key-value pair inside {{}} must be set as an individual property. Do not treat them as a single map.
        If a metric has multiple attributes (e.g., node_network_info{{address="02:42:db:56:1c:74", adminstate="up"}}), split them into separate properties like
           `MATCH (n:Network) SET n.address = "02:42:db:56:1c:74", n.adminstate = "up"`    
        Never use maps or JSON-like structures in Cypher queries. Each attribute should be a separate property.
        Do not approximate values, use the exact values provided.
        Batch updates: Set all properties in a single query per node, using multiple SET clauses when updating multiple properties at once.
        For multiple nodes, generate separate queries, separated by `;`.
        After each `MATCH` clause, there should be an `;` to separate the queries.
        
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
  
  logger.warning(f'----GENERATED METRICS: {' '.join(generated_metrics_script)}----')
  
  init_graph=True
  solution=cypher(problem='Metrics', cypher_script=' '.join(generated_metrics_script))
  return {'generation':solution, "messages" : messages, "iterations" : iterations} 
  
def reflect(state: GraphState):
  pass

def check_end(state: GraphState):
  global init_graph, metrics_component, components
  if init_graph:
    for label in node_labels:
      metrics_component[label].clear()
    for label in node_labels[:-1]:
      components[label].clear()  
  return init_graph

def check_existing_node(state: GraphState):
    global already_existing_node
    return already_existing_node



def inititialize_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node('node_existance', existing_node)
    workflow.add_node('component_gen', component_gen)
    workflow.add_node('validation_components', component_validation)
    workflow.add_node('validation_relationships', component_validation)
    workflow.add_node('validation_metrics', component_validation)
    workflow.add_node('relationship_gen', generate_relationships)
    workflow.add_node('metrics_gen', generate_metrics)
    workflow.add_node('run_script_components', run_script)
    workflow.add_node('run_script_relationships', run_script)
    workflow.add_node('run_script_metrics', run_script)
    workflow.add_node('update_node', update_node)
    
    workflow.add_edge(START, 'node_existance')
    workflow.add_conditional_edges('node_existance', check_existing_node, {True: 'update_node', False: 'component_gen'}) # checks if the node already exists
    workflow.add_edge('update_node', END)
    workflow.add_edge('component_gen', 'validation_components')
    workflow.add_edge('validation_components', 'run_script_components')
    workflow.add_edge('run_script_components', 'relationship_gen')
    workflow.add_edge('relationship_gen', 'validation_relationships')
    workflow.add_edge('validation_relationships', 'run_script_relationships')
    workflow.add_edge('run_script_relationships', 'metrics_gen')
    workflow.add_edge('metrics_gen', 'validation_metrics')
    workflow.add_edge('validation_metrics', 'run_script_metrics')
    
      
    workflow.add_conditional_edges('run_script_metrics', check_end, {True: END, False: 'metrics_gen'}) # reflect is only an empty node for now

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
  visualize_graph(app)
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
      
      For every node include their name as a property. For example, the `CPU` node with id `0` should have a property `name: "0"`. 
      
      {processed_components}
      '''
      )
  ]
  )
  messages = question.format_messages(processed_components=processed_components)
  if not isinstance(messages , list):
      messages = [messages]
  solution = app.invoke({"messages":messages, "iterations":0, "error":""})
  # components={label: set() for label in node_labels[:-1]}
  # metrics_component= {label: [] for label in node_labels}
  return solution

def query_graph(user_nl_query):
  print(graph.schema)
  return neo4j_chain.invoke({'query':user_nl_query})
  

#TODO implement the reflect system
#TODO implement the update node and relationship system
#TODO move all global variables to a main function and start the execution from there
#TODO chunk the metrics into smaller parts and process them in batches => the LLM has a token limit that could be reached