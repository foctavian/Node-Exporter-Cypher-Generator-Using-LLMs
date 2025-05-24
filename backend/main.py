from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import os
from contextlib import asynccontextmanager
import shutil
from agent import cypher_gen_chain
from pydantic import BaseModel
from agent_interface import AgentInterface
from datetime import datetime
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from typing import Set 
from file_manager import encode_file
import asyncio 

logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs.log', encoding='utf-8', level=logging.INFO)

scheduler = BackgroundScheduler()
main_event_loop=None
seen_files: Set[str] = set()
connected_clients=set()

async def async_broadcast_message(message: str):
    disconnected = []
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception as e:
            disconnected.append(client)
    
    for client in disconnected:
        connected_clients.remove(client)
        
def broadcast_message(message: str):
    global main_event_loop
    if main_event_loop:
        asyncio.run_coroutine_threadsafe(
            async_broadcast_message(message), main_event_loop
        )

#TODO move the processed files to a new directory

def check_for_new_files():
    global seen_files
    dr = 'uploads'
    try:
        current_files = set(os.listdir('uploads'))
        new_files = current_files - seen_files
        if new_files:
            print(f"New files detected: {new_files}")
            for f in new_files:
                nf=encode_file(os.path.join(dr, f))
                if nf:
                    broadcast_message(f"New file: {nf}") 
                seen_files.add(nf)
    except Exception as e:
        print(e)
 

@asynccontextmanager
async def lifespan(app:FastAPI):
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    scheduler.add_job(check_for_new_files, 'interval', seconds=20, id="file-manager") 
    scheduler.start()
    yield
    scheduler.shutdown()   
    
app = FastAPI(lifespan=lifespan)
class Message(BaseModel):
    question:str
    
class UpdateSystemNode(BaseModel):
    ip: str
    name: str
    timestamp: str

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

@app.websocket("/ws/notif")
async def notification_endpoint(websocket:WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception as e:
        print(e)

@app.post("/upload")
async def upload_metrics_file(file:UploadFile=File(...)):
    dir = 'uploads'
    if not os.path.exists(dir):
        os.makedirs(dir)
    file_location = os.path.join(dir, file.filename)

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            await AgentInterface().start_processing(file_location)
        return {'message': f"File {file.filename} was uploaded successfully"}
    except Exception as e:
        return {'error':f"File {file.filename} failed: {str(e)}"} 
    

@app.post("/agent-test")
async def test_question(data:Message):
    solution = cypher_gen_chain.invoke({'messages':[('user', data.question)]})
    return solution

@app.post("/query-graph") #TODO: fix error 429
async def query_graph(query: Message):
    try:
        response = await AgentInterface().query_graph(query.question)
        return {"cypher_query":response['intermediate_steps'], "result":response['result']}
    except Exception as _:
        import asyncio
        await asyncio.sleep(3)
        response = await AgentInterface().query_graph(query.question)
        return {"cypher_query":response['intermediate_steps'], "result":response['result']}

@app.get('/get-node-ips')
async def get_node_ips():
    dr = './uploads'
    node_ids = set()
    node_data = {}
    for f in os.listdir(dr):
        node_id = f.split('_')[0]
        node_timestamp = f.split('_')[2].split('.')[0] # get the timestamp
        node_ids.add(node_id)
        node_data[node_id] = node_timestamp
    return node_data
        
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

@app.post('/start-update')
async def start_update(node_to_update: UpdateSystemNode):
    scheduler.pause_job("file-manager")
    await AgentInterface().start_update(node_to_update.ip, node_to_update.name, node_to_update.timestamp)
    scheduler.resume_job("file-manager")

@app.get('/start-processing')
async def start_processing():
    AgentInterface().start_processing()
    

    
    
    
#TODO change their name and create a synced task that checks if there are any updates
