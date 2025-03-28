<script>
    import axios from 'axios';
    import { onMount } from "svelte";
    import { DataSet, Network } from "vis";
    import { writable } from "svelte/store";
    
    let graphContainer;
    let selectedNode = writable(null);
    let answer=writable('');
    let question='';
    async function fetchGraph() {
        try {
            const response = await axios.get("http://localhost:8000/get-current-graph");
            initializeVisGraph(response.data);
        } catch (error) {
            console.log(error);
        }
    }

    async function sendQuestion(){
        axios.post("http://localhost:8000/query-graph", {
            question: question
        },{headers: {
            "Content-Type": "application/json"
        }}).then(response => {
            answer=response.data.result;
        }).catch(error => {
            console.log(error);
        });
    }

    function initializeVisGraph(graphData) {
        let nodesMap = new Map();
        let edgeSet = new Set();
        let edges = [];

        graphData.forEach(({ source, target, relationship }) => {
            if (!nodesMap.has(source.id)) {
                nodesMap.set(source.id, { 
                    id: source.id, 
                    label: source.properties.id, 
                    group: source.labels[0], 
                    properties: source.properties 
                });
            }
            if (!nodesMap.has(target.id)) {
                nodesMap.set(target.id, { 
                    id: target.id, 
                    label: target.properties.id, 
                    group: target.labels[0], 
                    properties: target.properties 
                });
            }

            const edgeKey = [source.id, target.id].sort().join("-");
            if (!edgeSet.has(edgeKey)) {
                edges.push({
                    from: source.id,
                    to: target.id,
                    label: relationship.type,
                    arrows: 'to',
                    dashes: false
                });
                edgeSet.add(edgeKey);
            }
        });

        const data = {
            nodes: Array.from(nodesMap.values()),
            edges: edges
        };

        const options = {
            nodes: { shape: "dot", size: 15 },
            arrows: { to: { scaleFactor: 2 }, from: false },
            physics: { enabled: true },
            interaction: { hover: true }
        };

        const network = new Network(graphContainer, data, options);

        // Handle node click event
        network.on("click", function(params) {
            if (params.nodes.length > 0) {
                let nodeId = params.nodes[0];
                let node = nodesMap.get(nodeId);
                selectedNode.set(node);
            } else {
                selectedNode.set(null);
            }
        });
    }

    onMount(fetchGraph);
</script>

<style>
    #graph-container {
        width: 100%;
        display: flex;
        height: 85vh;
        border: 1px solid #ddd;
    }
    .sidebar {
        position: fixed;
        right: 0;
        top: 0;
        height: 100vh;
        width: 300px;
        background: #f9f9f9;
        border-left: 1px solid #ddd;
        padding: 20px;
        box-shadow: -2px 0 5px rgba(0, 0, 0, 0.1);
        overflow-y: auto;
        transition: transform 0.3s ease-in-out;
    }

    .hidden {
        transform: translateX(100%);
    }

    h3 {
        margin-top: 0;
    }
</style>

<div bind:this={graphContainer} id="graph-container"></div>
<div class="query-container">
    <input type="text" 
    bind:value={question} 
    placeholder="Query the database" />
    <button on:click={sendQuestion}>Send</button>
    <br>
    <h3>{answer}</h3>
</div>
<div class="sidebar {$selectedNode ? '' : 'hidden'}">
    {#if $selectedNode}
        <h3>Node {$selectedNode.properties?.name || $selectedNode.id} : {$selectedNode.group}</h3>
        <pre>{JSON.stringify($selectedNode.properties, null, 2)}</pre>
    {/if}
</div>