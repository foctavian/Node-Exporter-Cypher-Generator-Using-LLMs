from dotenv import load_dotenv
load_dotenv()
from paho.mqtt import client as mqtt_client
from .mqtt_client import get_client

def on_message(client, userdata, msg):
    print(f"{msg.topic}: {msg.payload.decode()}")

client = get_client()
client.on_message = on_message

client.subscribe("mesh/telemetry", qos=1)

client.publish("mesh/51/cmd/telemetry", "test")

import time
while True:
    time.sleep(1)