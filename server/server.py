from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import time
import os
import base64
import requests

PUSHBULLET_ENABLED = True

pushbulletKey = None
pushbulletMode = False
if PUSHBULLET_ENABLED and os.path.exists('./secret.txt'):
    with open('./secret.txt', 'r') as file:
        pushbulletMode = True
        pushbulletKey = file.read()

print("Pushbullet Mode:", str(pushbulletMode))

app = Flask(__name__)
CORS(app)

clients = []

names = ["Camera Name 1", "Camera Name 2", "Camera Name 3", "Camera Name 4",
         "Camera Name 5", "Camera Name 6", "Camera Name 7", "Camera Name 8"]
locations = ["Deep End North", "Diving Pool", "Deep End South", "Shallow End North",
             "Shallow End South", "North West Side", "South West Side", "North East Side"]


@app.get('/')
def index():
    return send_from_directory('dist/', 'index.html')


@app.post('/register/<threshold>')
def register(threshold):
    registeredId = len(clients)
    clients.append({
        "id": registeredId,
        "ip": request.remote_addr,
        "time": time.time(),
        "name": names[registeredId],
        "location": locations[registeredId],
        "threshold": float(threshold),
        "data": None,
    })
    print("Camera", str(registeredId), "registered at", request.remote_addr)
    return str(registeredId)


@app.post('/update/<id>')
def update(id):
    client = clients[int(id)]
    client['time'] = time.time()
    client['data'] = request.get_json()
    if pushbulletMode:
        if len(client["data"]["alerts"]) > 0:
            for alert in client["data"]["alerts"]:
                print("Sending Alert")
                message = f"Camera {client['name']} at {client['location']} has detected a swimmer that has been submerged for an extended period of time!"
                response = requests.post('https://api.pushbullet.com/v2/pushes', headers={
                                         'Access-Token': pushbulletKey, 'Content-Type': 'application/json'}, json={'type': 'note', 'title': 'SALMON ALERT', 'body': message})

    return "ok"


@app.get('/all')
def all():
    return jsonify(clients)


@app.get('/<file_name>/')
def get_file(file_name):
    return send_from_directory('dist/', file_name)


@app.get('/js/<file_name>/')
def get_js_file(file_name):
    return send_from_directory('dist/js/', file_name)


@app.get('/img/<file_name>/')
def get_img_file(file_name):
    return send_from_directory('dist/img/', file_name)


@app.get('/css/<file_name>/')
def get_css_file(file_name):
    return send_from_directory('dist/css/', file_name)


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app.run(host='0.0.0.0', port=80, use_reloader=False)
