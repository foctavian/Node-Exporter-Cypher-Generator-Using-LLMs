import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import '../style/Home.css';

const Home = () => {
    const [uploadMessage, setUploadMessage] = useState('No file has been uploaded.');
    const [selectedFile, setSelectedFile] = useState(null);
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [isDragActive, setIsDragActive] = useState(false);
   
    const navigate = useNavigate();

    const uploadFile = async () => {
      if (!selectedFile) return;
      setIsUploading(true);
      
      const formData = new FormData();
      if (selectedFile instanceof File) {
        formData.append("file", selectedFile);
      }
      
      try {
        const response = await axios.post("http://localhost:8000/upload", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        setUploadMessage(response.data.message);
      } catch (error) {
        setUploadMessage("Upload failed");
      } finally {
        setIsUploading(false);
      }
    };
    
    const startProcessing = async () => {
      setIsProcessing(true);
      try {
        await axios.get("http://localhost:8000/start-processing", {
          headers: { "Content-Type": "application/json" },
        });
        setUploadMessage("Processing completed");
      } catch (error) {
        setUploadMessage("Processing failed");
      } finally {
        setIsProcessing(false);
      }
    };
    
    const handleDragOver = (event) => {
      event.preventDefault();
      setIsDragActive(true);
    };
    
    const handleDragLeave = () => {
      setIsDragActive(false);
    };
    
    const dropFile = (event) => {
      event.preventDefault();
      setIsDragActive(false);
      if (event.dataTransfer?.files.length) {
        const file = event.dataTransfer.files[0];
        setSelectedFile(file);
        setUploadMessage(`Selected: ${file.name}`);
      }
    };
    
    const handleFileSelect = (event) => {
      const input = event.target;
      if (input.files && input.files.length > 0) {
        setSelectedFile(input.files[0]);
        setUploadMessage(`Selected: ${input.files[0].name}`);
      }
    };
    
    const removeSelectedFile = () => {
      setSelectedFile(null);
      setUploadMessage('No file has been uploaded.');
    };
    
    const sendQuestion = async () => {
      if (!question.trim()) return;
      
      try {
        const response = await axios.post("http://localhost:8000/query-graph", { question }, {
          headers: { "Content-Type": "application/json" },
        });
        setAnswer(response.data.cypher_script);
      } catch (error) {
        console.log(error);
        setAnswer("Error retrieving answer.");
      }
    };

  return (
    <div className="dark-theme">
      <div className="file-upload-container">
        <h1 className="page-title">{uploadMessage}</h1>
        
        <div
          className={`drop-zone ${isDragActive ? 'active' : ''}`}
          role="form"
          onDrop={dropFile}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div className="drop-zone-icon">📄</div>
          <p className="drop-zone-text">Drag & Drop a file here</p>
          
          <div className="file-input-wrapper">
            <label className="file-input-label">
              Browse Files
              <input
                type="file"
                onChange={handleFileSelect}
                accept=".txt"
                className="file-input"
              />
            </label>
          </div>
          
          {selectedFile && (
            <div className="selected-file">
              <span className="file-name">{selectedFile.name}</span>
              <button onClick={removeSelectedFile} className="remove-file">×</button>
            </div>
          )}
        </div>
        
        <button 
          onClick={uploadFile} 
          disabled={!selectedFile || isUploading} 
          className={`action-btn ${isUploading ? 'loading' : ''}`}
        >
          {isUploading ? 'Uploading...' : 'Upload File'}
        </button>
        
        
        <div className="button-group">
          <button onClick={() => navigate('/graph')} className="navigation-btn">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16" style={{ marginRight: '8px' }}>
              <path fillRule="evenodd" d="M0 0h1v15h15v1H0V0Zm14.817 3.113a.5.5 0 0 1 .07.704l-4.5 5.5a.5.5 0 0 1-.74.037L7.06 6.767l-3.656 5.027a.5.5 0 0 1-.808-.588l4-5.5a.5.5 0 0 1 .758-.06l2.609 2.61 4.15-5.073a.5.5 0 0 1 .704-.07Z"/>
            </svg>
            View Graph
          </button>
          
          <button 
            onClick={startProcessing} 
            className={`processing-btn ${isProcessing ? 'loading' : ''}`}
            disabled={isProcessing}
          >
            {isProcessing ? 'Processing...' : 'Start Processing'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Home;
