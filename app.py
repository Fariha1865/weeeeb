
import numpy as np
from flask import Flask, request, jsonify, render_template,Response
import pickle
import cv2
import time
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils

import dlib

import os
import pyrebase
from firebase import firebase
import pygame
pygame.mixer.init()
time.sleep(1)
app = Flask(__name__)
config = {
   "apiKey": "AIzaSyC40LmBxIdvaUZRPf8cP9jCIx_QJqu4DFY",
  "authDomain": "oversee-70b41.firebaseapp.com",
  "databaseURL": "https://oversee-70b41-default-rtdb.firebaseio.com",
  "projectId": "oversee-70b41",
  "storageBucket": "oversee-70b41.appspot.com",
  "messagingSenderId": "62841252291",
  "appId": "1:62841252291:web:0f7fe37c37beb4c73bb576",
  "measurementId": "G-D0XNDG0REX"
}
firebase = firebase.FirebaseApplication("https://console.firebase.google.com/u/0/project/oversee-6fa5e/database/oversee-6fa5e-default-rtdb/data/~2FTestVal")
firebase_db = pyrebase.initialize_app(config)
db = firebase_db.database()
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance




@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('page2.html')

@app.route('/back')
def back():
    return render_template('index.html')
@app.route('/fire')
def fire():
    return render_template('this is new.html')

@app.route('/next')
def next():
    return render_template('state.html')
@app.route('/previous')
def previous():
    return render_template('page2.html')
@app.route('/meh/')
def meh():
    return render_template('meh.html')
@app.route('/my-link/')
def my_link():
  ap = argparse.ArgumentParser()
  ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
  args = vars(ap.parse_args())

  print("-> Loading the predictor and detector...")
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

  print("-> Starting Video Stream")
  vs = VideoStream(src=args["webcam"]).start()
  time.sleep(1.0)

  frame = vs.read()
  frame = imutils.resize(frame, width=450)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  rects = detector(gray, 0)

  for rect in rects:
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    distance = lip_distance(shape)
    lip = shape[48:60]
    cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

  db.child("TestVal").update({"lip":distance})
  return "done"

if __name__ == "__main__":

  app.run(port='8000',debug=True)

  