from flask import Flask, request, jsonify, send_from_directory
import logging
import time

app = Flask(__name__)

clients = []

names = ["Camera Name 1", "Camera Name 2", "Camera Name 3", "Camera Name 4", "Camera Name 5", "Camera Name 6", "Camera Name 7", "Camera Name 8"]
locations = ["Deep End North", "Diving Pool", "Deep End South", "Shallow End North", "Shallow End South", "North West Side", "South West Side", "North East Side"]

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
    return str(registeredId)

@app.post('/update/<id>')
def update(id):
    clients[int(id)]['time'] = time.time()
    clients[int(id)]['data'] = request.get_json()
    return "ok"

@app.get('/all')
def all():
    return jsonify(clients)

@app.get('/<file_name>/')
def get_file(file_name):
    return send_from_directory('dist/', file_name)

#Serve static Vue files
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