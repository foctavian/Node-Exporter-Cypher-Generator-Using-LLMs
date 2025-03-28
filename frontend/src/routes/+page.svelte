<script>
    import axios from 'axios';
    import { writable } from "svelte/store";
    import { goto } from '$app/navigation';

    let title = 'NEO4J LLM DEMO APP'
    let uploadMessage = writable('No file has been uploaded.')
    let selectedFile= null;
    let question='';
    let answer=writable('');

    async function uploadFile(){
        const formData=new FormData();
        if(selectedFile instanceof File){
            formData.append("file", selectedFile);
        }
        
        try{
            const response = await axios.post("http://localhost:8000/upload", formData,{
                headers:{"Content-Type":"multipart/form-data"}
            });
            uploadMessage.set(response.data.message)
        }catch(error){
            uploadMessage.set("Upload failed");
        }
    }

    async function startProcessing(){
        try{
            const response = await axios.get("http://localhost:8000/start-processing", {},
            {headers: {"Content-Type": "application/json"}});
        }catch(error){
            uploadMessage.set("Processing failed");
        }
    }

    async function dropFile(event){
        event.preventDefault();
        if (event.dataTransfer?.files.length){
            selectedFile = event.dataTransfer.files[0];
        }
    }

    function handleDragOver(event) {
        event.preventDefault();
    }

    function handleFileSelect(event) {
        const input = event.target;
        if (input.files && input.files.length > 0) {
            selectedFile = input.files[0];
            uploadMessage.set(`Selected: ${selectedFile.name}`);
        }
    }

    async function sendQuestion(){
        axios.post("http://localhost:8000/query-graph", {
            question: question
        },{headers: {
    "Content-Type": "application/json"
}}).then(response => {
            answer=response.data.cypher_script;
        }).catch(error => {
            console.log(error);
        });
    }

</script>

<style>
        .drop-zone {
        border: 2px dashed #ccc;
        padding: 20px;
        text-align: center;
        cursor: pointer;
        width: 100%;
        height: 150px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .drop-zone:hover {
        border-color: #3498db;
    }
</style>

<title>{title}</title>
<h1>{$uploadMessage}</h1>

<div class="drop-zone" role="form" on:drop={dropFile} on:dragover={handleDragOver}>
    Drag & Drop a file here or 
    <br>
    <input type='file' on:change={handleFileSelect} accept=".txt"/>
</div>

<button on:click={uploadFile} disabled={!selectedFile}>
    Upload file
</button>
<br/>

<div>
    <input type='text' bind:value={question} />
</div>
<button on:click={sendQuestion}>
    Send question
</button>
<br/>
<div>
    <p>ANSWER: {answer}</p>
</div>

<button on:click={() => goto('/graph')}>
    Go to graph
</button>
<br/>
<button on:click={startProcessing}>
    Start processing
</button>