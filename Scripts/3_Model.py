import pandas as pd
import numpy as np
import joblib
import librosa
import os
from pathlib import Path


## Audio Processing

def extract_features(audio_file):
  y, sr = librosa.load(audio_file, sr=16000)
  q25, q75 = np.percentile(y, [25, 75])
  iqr = q75 - q25
  meanfun = np.mean(librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512))

  return {
    "q25": q25,
    "iqr": iqr,
    "meanfun": meanfun,
  }

audio_file = input("Enter wav <filename> without extension: ")+'.wav'
audio_file = Path(__file__).parent.parent / 'AudioFiles' / audio_file

if not os.path.isfile(audio_file):
  print(f"Error: WAV file '{audio_file}' not found!")
  exit()

features = {}
# try:
  # features = extract_features(audio_file)
# except Exception as e:
  # print(f"An error occurred while processing '{audio_file}': {e}")
  # exit()
features = extract_features(audio_file)

X = pd.DataFrame({"q25":[features['q25']],"iqr":[features['iqr']],"meanfun":[features['meanfun']]})


## Prediction

label_encode = joblib.load("label_encode.pkl")

model=joblib.load('models/log_reg.pkl')
print("Log Reg:\t",list(label_encode.inverse_transform(model.predict(X)))[0])

model=joblib.load('models/svm.pkl')
print("SVM:\t\t",list(label_encode.inverse_transform(model.predict(X)))[0])

model=joblib.load('models/random_forest.pkl')
print("Random forest:\t",list(label_encode.inverse_transform(model.predict(X)))[0])

model=joblib.load('models/knn.pkl')
print("KNN:\t\t",list(label_encode.inverse_transform(model.predict(X)))[0])
