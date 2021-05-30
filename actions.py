import re, operator
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def check_elem_in_string(arr, str_arr):
    for elem in arr:
        if elem in str_arr:
            return True
    return False

def save_data(data):
    file = open('saved_data.json', 'w')
    json.dump(data, file, indent=4)

"""
Steps to prediction:
    1. Collect user liked data (raw) & convert it into something evaluable (preparation)
    2. Initialize the 500,000 songs csv file, read it, extract popular songs from it
    3. Apply model logic to make clusters of those popular songs & get the cluster number for user liked songs
    4. Get songs from the popular songs dataframe with the same cluster number
"""

# 1
# Take data sent on request and converts it into panda dataframe
def prepare_liked_data(raw_data):
    user_data = []
    feature_names = ['id', 'danceability', 'energy', 'loudness',
                    'tempo', 'acousticness', 'valence', 'speechiness',
                    'instrumentalness', 'liked_by_user']

    for playlist_id, tracks in raw_data:
        for track in tracks:
            user_data.append([
                track['id'], track['danceability'], track['energy'],
                track['loudness'], track['tempo'], track['acousticness'],
                track['valence'], track['speechiness'], track['instrumentalness'], 1
            ])

    return pd.DataFrame(user_data, columns=feature_names)

# 2
# reads data from 500,000 song dataset and gets only songs which are popular enough
# reason for choosing popularity was because the dataset had very old and forgotten songs also
# moreover, applying this constraint also helps in predictions as:
#   if prediction is wrong, at least the song is tolerable <:)
def read_historical_data():
    file_tracks = pd.read_csv('./tracks.csv')

    populars = []
    for row in np.array(file_tracks):
        if row[2] > 75: populars.append(row) # only add songs who have popularity > 75

    popular_tracks = pd.DataFrame(populars, columns=['id', 'name', 'popularity', 'duration_ms', 'explicit', 'artists',
        'id_artists', 'release_date', 'danceability', 'energy', 'key',
        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'time_signature'])
    
    return popular_tracks

# 3
# Garbage logic applied here:
#   Clusterized whole popular_tracks dataframe into 3 clusters (based on energy: high, medium, low) along with liked_tracks
#   Obtained the most prominent cluster number
#   get songs with the obtained cluster number
def make_predictions(raw_data):
    popular_tracks = read_historical_data()
    liked_tracks = prepare_liked_data(raw_data)

    prediction_factors = ['energy', 'loudness', 'tempo', 'acousticness', 'valence']

    # ml model initialization, using KMeans algorithm
    model = KMeans(n_clusters=3).fit(
        popular_tracks[prediction_factors]
    )

    liked_cluster = round(model.predict(liked_tracks[prediction_factors]).mean())
    popular_tracks['cluster_no'] = model.predict(popular_tracks[prediction_factors])

    # Get most listened artists
    liked_artists = {}
    for row in np.array(popular_tracks[['artists', 'cluster_no']]):
        if row[1] != liked_cluster:
            continue

        artists = row[0].replace('[', '').replace(']', '').replace("'", "").replace("\"", "")
        if ',' in artists:
            artist_arr = artists.split(', ')
        else: artist_arr = [artists]
        
        for artist in artist_arr:
            liked_artists[artist] = liked_artists.get(artist, 0) + 1

    liked_artists = dict([i for i in sorted(liked_artists.items(), key=operator.itemgetter(1), reverse=True) if i[1] > 2])

    recommended = []
    for row in np.array(popular_tracks[['name', 'artists', 'id', 'cluster_no']]):
        if row[-1] == liked_cluster and check_elem_in_string(liked_artists.keys(), row[1]):
            recommended.append([row[0], row[2]])
    
    return recommended
