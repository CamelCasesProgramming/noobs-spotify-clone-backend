from flask import Flask, request, Response
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
import json, time, random, os
from actions import *

app = Flask(__name__)
cors = CORS(app, resource={
    r"/*": {
        "origins": "*"
    }
})


# Checks if the server is running or not
@app.route('/', methods=['GET'])
def check():
    return {'status': True, 'message': 'Running server'}


# Gets the data of songs which user likes, to fit, train and predict on it
@app.route('/push-user-friendlies', methods=['POST'])
def push_user_friendlies():
    if (len(request.json) < 1): return { 'status': False, 'message': 'You need to be a regular user of spotify in order to get recommendations from us :)' }

    recommendations = make_predictions(request.json)
    recom_tracks = set()
    while len(recom_tracks) < 50:
        recom_tracks.add(recommendations[random.randint(0, len(recommendations) - 1)][1])

    return {'status': True, 'recommendations': list(recom_tracks)}

try:
    print('Running server')
    http_server = WSGIServer(('', int(os.environ.get('PORT'))), app)
    http_server.serve_forever()

except Exception as err:
    print("\n\nError occured while serving\n\n")
    print(err)
