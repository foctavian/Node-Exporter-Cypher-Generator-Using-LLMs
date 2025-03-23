from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from agent import cypher_gen_chain
from pydantic import BaseModel
from agent_interface import AgentInterface
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs.log', encoding='utf-8', level=logging.INFO)

app = FastAPI()
class Message(BaseModel):
    question:str

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/upload")
async def upload_metrics_file(file:UploadFile=File(...)):
    dir = 'uploads'
    if not os.path.exists(dir):
        os.makedirs(dir)
    file_location = os.path.join(dir, file.filename)
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            AgentInterface().start_processing(file_location)
        return {'message': f"File {file.filename} was uploaded successfully"}
    except Exception as e:
        return {'error':f"File {file.filename} failed: {str(e)}"} 
    

@app.post("/agent-test")
async def test_question(data:Message):
    solution = cypher_gen_chain.invoke({'messages':[('user', data.question)]})
    return solution

@app.post("/query-graph")
async def query_graph(query: Message):
    return AgentInterface().query_graph(query.question)
    

@app.get('/get-current-graph')
async def get_current_graph():
    records, summary, keys = AgentInterface().retrieve_current_graph()

    graph_data = []

    for record in records:
        node1 = record['N']
        rel = record['R']
        node2 = record['M']

        graph_data.append({
            "source": {"id": node1.id, "labels": list(node1.labels), "properties": dict(node1)},
            "target": {"id": node2.id, "labels": list(node2.labels), "properties": dict(node2)},
            "relationship": {"type": rel.type, "properties": dict(rel)}
        })

    return graph_data

@app.get('/start-processing')
async def start_processing():
    AgentInterface().start_processing()
    