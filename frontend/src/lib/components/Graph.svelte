<script>
    import axios from 'axios';
    import { onMount } from "svelte";
    import { DataSet, Network } from "vis";
    import { writable } from "svelte/store";

    let graphContainer;
    let selectedNode = writable(null);

    async function fetchGraph() {
        try {
            const response = await axios.get("http://localhost:8000/get-current-graph");
            initializeVisGraph(response.data);
        } catch (error) {
            console.log(error);
        }
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
        height: 400px;
        border: 1px solid #ddd;
    }
</style>

<div bind:this={graphContainer} id="graph-container"></div>

<div>
    <h3>{$selectedNode ? `Node ${$selectedNode.properties?.name || $selectedNode.id} : ${$selectedNode.group}` : "No node selected"}</h3>
    {#if $selectedNode}
        <pre>{JSON.stringify($selectedNode.properties, null, 2)}</pre>
    {:else}
        <p>Click on a node to see details.</p>
    {/if}
</div>
