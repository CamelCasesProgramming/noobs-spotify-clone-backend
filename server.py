from flask import Flask, request, Response
from flask_cors import CORS
import json, time, random
from actions import *

app = Flask(__name__)
cors = CORS(app)


# Checks if the server is running or not
@app.route('/check', methods=['GET'])
def check():
    return {'status': True, 'message': 'Running server'}


# Gets the data of songs which user likes, to fit, train and predict on it
@app.route('/push-user-friendlies', methods=['POST'])
def push_user_friendlies():
    recommendations = make_predictions(request.json)

    recom_tracks = set()
    while len(recom_tracks) < 50:
        recom_tracks.add(recommendations[random.randint(0, len(recommendations) - 1)][1])

    return {'status': True, 'recommendations': list(recom_tracks)}


app.run(debug=True)
