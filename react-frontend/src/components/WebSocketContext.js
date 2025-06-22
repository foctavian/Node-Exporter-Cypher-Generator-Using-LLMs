import React, { createContext, useContext, useEffect, useRef, useState } from "react";

const WebSocketContext = createContext();

export const WebSocketProvider = ({ children }) => {
  const socketRef = useRef(null);
  const [socketReady, setSocketReady] = useState(false);
  const [notifications, setNotifications] = useState('');
  const [logs, setLogs]= useState([]);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");

    ws.onopen = () => {
      console.log("WebSocket connected");
      setSocketReady(true);
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
      setSocketReady(false);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if(data.type==='notification'){
        setNotifications(event.data.message);
      }
      else{
        console.log("Received message:", data);
        setLogs((prevLogs) => [...prevLogs, data.message]);
      }
    };

    ws.onerror = (err) => {
      console.error("WebSocket error", err);
    };

    socketRef.current = ws;

    return () => {
      ws.close();
    };
  }, []);

  const sendMessage = (msg) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(msg);
    } else {
      console.warn("WebSocket not connected");
    }
  };

  return (
    <WebSocketContext.Provider value={{ socket: socketRef.current, sendMessage, socketReady, notifications, logs }}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error("useWebSocket must be used within a WebSocketProvider");
  }
  return context;
};
