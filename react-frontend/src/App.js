import logo from './logo.svg';
import './App.css';
import '@radix-ui/themes/styles.css';
import { Flex, Text, Button, Theme } from "@radix-ui/themes";
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Graph from './components/Graph';
import Home from './pages/Home';
import { useState } from 'react';
import { useEffect } from 'react';
function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />}/>
        <Route path="/graph" element={<Graph />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
