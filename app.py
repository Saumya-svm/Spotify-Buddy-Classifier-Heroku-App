import streamlit as st
import time
import spotipy
import spotipy.util as util
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,OneHotEncoder
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import time
from collections import Counter
from imblearn.over_sampling import SMOTE
import sys
import gspread


# importing python file containing my credentials for using Spotify API
import credentials
username = credentials.username
client_id = credentials.client_id
client_secret = credentials.client_secret
redirect_uri = 'http://localhost:https://spotifybuddyclassifier.herokuapp.com//callback'
scope = 'user-read-recently-played'

token = util.prompt_for_user_token(username=username, 
                                   scope=scope, 
                                   client_id=client_id,   
                                   client_secret=client_secret,     
                                   redirect_uri=redirect_uri)
sp = spotipy.Spotify(auth=token)

def get_playlist_tracks(playlist_id):
    results = sp.user_playlist_tracks(username,playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks


def get_features(track_id: str, token: str) -> dict:
        sp = spotipy.Spotify(auth=token)
        try:
            features = sp.audio_features([track_id])
            return features
        except:
            return None
        

def return_playlist(playlist_id):
    tracks = get_playlist_tracks(playlist_id)

    l = []
    for i in tracks:
        l.append(i['track']['uri'].split(':')[-1])
    string_list = []
    for i in range(len(tracks)//100 + 1):
        string_list.append(','.join(l[i*100:(i+1)*100]))

    json_list = []
    for s in string_list:
        headers = {'Accept': 'application/json','Content-Type': 'application/json','Authorization': f'Bearer ' + token}
        params = [('ids', s)]
        json = ''
        try:
            response = requests.get('https://api.spotify.com/v1/audio-features', 
                        headers = headers, params = params, timeout = 5)
            json = response.json()
            json_list.append(json)
        except:
            print('None')
    dataframe_list = []
    for json in json_list:
        try:
            dataframe_list.append(pd.DataFrame(json['audio_features']))
        except:
            st.write('🚨🚨 Local files detected. Upload tracks from Spotify and press R to rerun.')
            sys.exit()
    df = pd.concat(dataframe_list)
    return df
        
def clean_data(a_data,b_data,class2,class1='Saumya Mundra'):
    a_data['class'] = class1
    b_data['class'] = class2
    

    feature_list = [ 'acousticness',
           'danceability', 'duration_ms', 'energy', 'instrumentalness', 'key',
           'liveness', 'loudness', 'mode', 'speechiness', 'tempo',
           'time_signature', 'valence','class']
    
    a_features = a_data[feature_list]
    b_features = b_data[feature_list]

    df = pd.concat([a_features,b_features])
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    #print(Counter(y))
    smote = SMOTE()
    X,y = smote.fit_resample(X,y)
    #print(Counter(y))
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(X_train[['tempo']])

    X_train['tempo'] = scaler.transform(X_train[['tempo']])
    X_test['tempo'] = scaler.transform(X_test[['tempo']])

    return X_train,X_test,y_train,y_test

def train_model(X_train,X_test,y_train,y_test,k=13):
    model = RandomForestClassifier(random_state=1)
    fs = SelectKBest(score_func=f_classif,k=k)
    X_train_final = pd.DataFrame(fs.fit_transform(X_train,y_train),columns = X_test.columns[fs.get_support(indices=True)])
    X_test_final = X_test[X_test.columns[fs.get_support(indices=True)]]
    model.fit(X_train_final,y_train)
    return model


def make_prediction(song_id,feature_list,model):
    song = pd.DataFrame(get_features(track_id=song_id,token=token))[feature_list]
    final_prediction = model.predict(song)
    if final_prediction:
        return final_prediction

def friends_comparison(playlist_id1,playlist_id2,song_id,name1,name2):
    streamlit1 = return_playlist(playlist_id1)
    streamlit2 = return_playlist(playlist_id2)
    
    a_data = streamlit1
    b_data = streamlit2
    a_data = a_data[:-1]
    b_data = b_data[:-1]
    
    X_train,X_test,y_train,y_test = clean_data(a_data,b_data,name2,name1)
    feature_list = X_train.columns
    model = train_model(X_train,X_test,y_train,y_test)

    #song_id = '0ofHAoxe9vBkTCp2UQIavz'
    prediction = make_prediction(song_id,feature_list,model)
    return prediction


def default_comparison(playlist_id,song_id,name):
    streamlit = return_playlist(playlist_id)
    a_data = pd.read_csv('model_playlist.csv')
    b_data = streamlit

    a_data = a_data.drop('Unnamed: 0',axis = 1)
    #b_data = b_data.drop('Unnamed: 0',axis = 1)
    b_data = b_data[:-1]
    
    X_train,X_test,y_train,y_test = clean_data(a_data,b_data,name)
    feature_list = X_train.columns
    model = train_model(X_train,X_test,y_train,y_test)

    #song_id = '0ofHAoxe9vBkTCp2UQIavz'
    prediction = make_prediction(song_id,feature_list,model)
    return prediction
    
def none_radio(list):
    st.write(f'👆🏻👆🏻 Kindly choose between {list[1]} and {list[2]} ')
    
def extract_id(link):
    try:
        id = link[:link.index('?')]
        return id
    except:
        return link

def run_streamlit():
    with st.sidebar:
        st.markdown(' ## Reference Links')
        st.write('[Spotipy](https://spotipy.readthedocs.io/)')
        st.write('[Spotify API](https://developer.spotify.com/documentation/web-api/)')
        st.write('[Spotify Web](https://open.spotify.com/)')
        st.markdown('## Contact Me')
        st.write('[Linkedin](https://www.linkedin.com/in/saumya-mundra/)')
        st.write('[Github](https://github.com/Saumya-svm)')
        
        
    
    st.header('Spotify Buddy Classifier')
    st.write("""
    This buddy classifier tells you who is more likely to listen to a particular song, you or your friend. You can also check who is more likely to listen to the song between you and me, the developer of this web app.
    
    - To run the classifier, we will need data, which you can provide through your Spotify Playlist Links. A text bar will be provided where the user (you or your friend) can enter their Spotify playlist link.
    
    - Once you provide your Spotify Playlist Link, this app will access the [Spotify API](https://developer.spotify.com/documentation/web-api/) and use ['Spotipy'](https://spotipy.readthedocs.io/), a python library for Spotify Web API, to get the track audio features such as tempo, danceability, acousticness etc. 
    
    - Once we have features corresponding to you and your friend's(or my) music taste, we will train the model. 
    
    - Then the trained model will classify which of the two users will be more likely to listen to the song.
    
    
    """)
    
    options = ['None','Me (To compare with me, choose this option)','Friend (To compare with friend, choose this option)']
    response = st.radio('',options,index=0)
    
    if response == options[1]:
        name = st.text_input('Type your Name Here',placeholder='Name')
        playlist_link = extract_id(st.text_input('Enter Playlist Link',placeholder='Link').strip().split('/')[-1])
        song_link = extract_id(st.text_input('Enter Song Link',placeholder='Link').split('/')[-1])
        if playlist_link and song_link:
            prediction = default_comparison(playlist_link,song_link,name)
            if prediction:
                st.write(f"""
                {prediction[0]} is more likely to listen to the new song.
                
                Success!
                
                Press R to Rerun.
                """)
    
    if response == 'None':
        none_radio(['None','Model','Friend'])
    
    if response == options[2]:
        name1 = st.text_input('Type your Name here',placeholder='Name')
        playlist_link1 = extract_id(st.text_input('Enter Your Playlist Link',placeholder='Link').strip().split('/')[-1])
        name2 = st.text_input("Type your Friend's Name here",placeholder='Name')
        playlist_link2 = extract_id(st.text_input("Enter Your Friend's Playlist Link",placeholder='Link').strip().split('/')[-1])
        song_link = extract_id(st.text_input('Enter Song Link',placeholder='Link').split('/')[-1])
        if playlist_link1 and playlist_link2 and song_link:
            prediction = friends_comparison(playlist_link1,playlist_link2,song_link,name1,name2)
            if prediction:
                st.write(f"""
                {prediction[0]} is more likely to listen to the new song.
                
                Success!
                
                Press R to Rerun.
                """)
                
    
    with st.form(key='message_form'):
        st.markdown('## Send a Message')
        l = []
        form_submission = pd.DataFrame(columns=['Name','Email','Message'])
        name_input = st.text_input(label='Name',placeholder='Name')
        email_input = st.text_input(label='Email',placeholder='email')
        message = st.text_area(label='Message',placeholder='Message')
        submit_button = st.form_submit_button(label='Submit')
        dict = {}
        if submit_button:
            f = open('credentials.json')
            credentials = json.load(f)
            gc, authorized_user = gspread.oauth_from_dict(credentials,{'refresh_token': '1//0g2h2Vn0uwzBGCgYIARAAGBASNwF-L9IrUR8sAzQgXXcg-ffc28Z2gNu7yT15mH27kzRcKk5F_fgVG28JoQkFWV5Hq5FfTi6vMKk',
 'token_uri': 'https://oauth2.googleapis.com/token',
 'client_id': '1054144430029-1hker5iq86m89h1p2ote3j45ea2d7513.apps.googleusercontent.com',
 'client_secret': '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',
 'scopes': ['https://www.googleapis.com/auth/spreadsheets',
  'https://www.googleapis.com/auth/drive'],
 'expiry': '2022-05-20T18:51:43Z'})
            authorized_user = json.loads(authorized_user)
            sh = gc.open_by_key("1SarcN0JtXQ5lUIylZC-_vTuYeEhCv1XmDi0dxXZKHew")
            sheet = sh.get_worksheet(0)
            dict['Name'] = name_input
            dict['Email'] = email_input
            dict['Message'] = message
            l = list(dict.values())
            sheet.append_row(l)
            form_submission = form_submission.append(dict,ignore_index=True)
            form_submission.to_csv('form_submissions.csv',mode='a',header=False)
run_streamlit()