import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import { Network } from "vis-network";
import { DataSet } from "vis-data";
import "vis-network/styles/vis-network.css";
import "../style/Graph.css";

const Graph = () => {
  const graphContainerRef = useRef(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const [answer, setAnswer] = useState("");

  useEffect(() => {
    async function fetchGraph() {
      try {
        const response = await axios.get(
          "http://localhost:8000/get-current-graph"
        );
        initializeVisGraph(response.data);
      } catch (error) {
        console.error(error);
      }
    }
    fetchGraph();
  }, []);

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendQuestion();
    }
  };

  const sendQuestion = async () => {
    setMessages((old) => [...old, { role: "user", content: question }]);
    setQuestion("");
    try {
      const response = await axios.post(
        "http://localhost:8000/query-graph",
        { question },
        { headers: { "Content-Type": "application/json" } }
      );
      setMessages((old) => [
        ...old,
        { role: "bot", content: response.data.result },
      ]);
    } catch (error) {
      console.error(error);
    }
  };

  const initializeVisGraph = (graphData) => {
    const nodesMap = new Map();
    const edgeSet = new Set();
    const edges = [];

    graphData.forEach(({ source, target, relationship }) => {
      if (!nodesMap.has(source.id)) {
        nodesMap.set(source.id, {
          id: source.id,
          label: source.properties.id,
          group: source.labels[0],
          properties: source.properties,
        });
      }
      if (!nodesMap.has(target.id)) {
        nodesMap.set(target.id, {
          id: target.id,
          label: target.properties.id,
          group: target.labels[0],
          properties: target.properties,
        });
      }

      const edgeKey = [source.id, target.id].sort().join("-");
      if (!edgeSet.has(edgeKey)) {
        edges.push({
          from: source.id,
          to: target.id,
          label: relationship.type,
          arrows: "to",
          dashes: false,
        });
        edgeSet.add(edgeKey);
      }
    });

    const data = {
      nodes: new DataSet(Array.from(nodesMap.values())),
      edges: new DataSet(edges),
    };

    const options = {
      nodes: { shape: "dot", size: 15 },
      arrows: { to: { scaleFactor: 2 }, from: false },
      physics: { enabled: true },
      interaction: { hover: true },
    };

    const network = new Network(graphContainerRef.current, data, options);

    network.on("click", function (params) {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        const node = nodesMap.get(nodeId);
        setSelectedNode(node);
      } else {
        setSelectedNode(null);
      }
    });
  };

  return (
    <div className="app-container">
      <div className="main-row">
        <div ref={graphContainerRef} id="graph-container"></div>
        <div className={`sidebar ${selectedNode ? "" : "hidden"}`}>
          {selectedNode && (
            <>
              <h3>
                Node {selectedNode.properties?.name || selectedNode.id} :{" "}
                {selectedNode.group}
              </h3>
              <pre>{JSON.stringify(selectedNode.properties, null, 2)}</pre>
            </>
          )}
        </div>
      </div>
      <div className="chat-container">
        {/* Chat Messages */}
        <div className="chat-history">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.role}-message`}>
                <div
                className="message-content"
                >
              <strong >{message.role === "user" ? "User" : "Bot"}:
                </strong>{" "}
                {message.content}
                </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyUp={handleKeyPress}
            placeholder="Ask about the graph..."
          />
          <button onClick={sendQuestion} disabled={!question.trim()}>
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default Graph;
