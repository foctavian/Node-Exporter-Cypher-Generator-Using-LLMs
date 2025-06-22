###################################### Insert generation ###############################

node_generation_prompt = f'''
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

generate_relationship_script_prompt = f'''
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

metric_generation_prompt = f'''
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
    {metrics_data}
'''


######################################## Extraction #######################################

node_extraction_prompt = f'''
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

relationships_extraction_prompt = f'''
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

metrics_extraction_prompt=f'''
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

################################ Update generation #####################################

generate_node_update_script_prompt = f'''

    Convert these components into Cypher statements to create nodes using `MERGE` only. 
    Do not use `CREATE`.

    For **each** node, use the following pattern:
    ```
    MERGE (var:Label {{system: "system_name", name: "node_name"}})
    ON CREATE SET var.createdAt = timestamp()
    ON MATCH SET var.updatedAt = timestamp()
    ```

    - Variable names should follow the format: `label_idOrName` (e.g., `cpu_0`)
    - Do not use special characters like `-` in variable names
    - Add a property `system: "{processed_components['NODENAME'][0]}"` to every node **except** the `NODENAME` node
    - Always include `name: "..."` as a property on the node
    - Do **not** return any output values in the final query
    - The `NODENAME` node should not have a system property

    For context, the graph represents a computation unit with various hardware components.

    Input:
    {missing_nodes_dict}

'''

generate_relationhip_update_script_prompt = f'''
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
'''

generate_metric_update_script_prompt = f'''
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
''' 