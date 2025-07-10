import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## Dataset Analysis

print("\n\nLoading dataset")

df = pd.read_csv('voice.csv')


print("\n\nDataset description:")

def describe(head=0,shape=0,info=0,description=0):
  if head==1:
    print("\nHead:")
    print(df.head(10))

  if shape==1:
    print("\nShape:")
    print(df.shape)

  if info==1:
    print("\nInfo:")
    df.info()

  if description==1:
    print("\nDescription:")
    print(df.describe())

describe(1,1,1,1)


## Encoding

print("\n\nEncoding:")

from sklearn.preprocessing import LabelEncoder
import joblib
label_encoder=LabelEncoder()

df["gender_encoded"]=label_encoder.fit_transform(df["gender"])
joblib.dump(label_encoder,'label_encode.pkl')

df=df.drop(labels="gender",axis=1)
describe(head=1,info=1)


## Data Cleaning

print("\n\nMissing values per column:")

missing_values = df.isnull().sum()
print(missing_values)


print("\n\nNumber of outliers in each column:")

z_scores = (df - df.mean()) / df.std()
threshold = 4
outliers = np.abs(z_scores) > threshold
num_outliers = outliers.sum()
print(num_outliers)

num_outliers_per_row = outliers.sum(axis=1)

print("\nOriginal DataFrame:")
describe(shape=1,description=1)

df = df[num_outliers_per_row == 0]

print("\nCleaned DataFrame:")
describe(shape=1,description=1)


## Principle Component Analysis (PCA)

print("\n\nFinding out most significant features:")

df2 = df.iloc[:,0:10]
fig,axes = plt.subplots(2,5,figsize=(20,10))
for k in range(1,11):
  ax = plt.subplot(2,5,k)
  sns.kdeplot(df2.loc[df['gender_encoded'] == 0, df2.columns[k-1]], color= 'green', label='F')
  sns.kdeplot(df2.loc[df['gender_encoded'] == 1, df2.columns[k-1]], color= 'red', label='M')
plt.show()

df2 = df.iloc[:,10:20]
fig,axes = plt.subplots(2,5,figsize=(20,10))
for k in range(1,11):
  ax = plt.subplot(2,5,k)
  sns.kdeplot(df2.loc[df['gender_encoded'] == 0, df2.columns[k-1]], color= 'green', label='F')
  sns.kdeplot(df2.loc[df['gender_encoded'] == 1, df2.columns[k-1]], color= 'red', label='M')
plt.show()

# exit()

print("Selecting most significant features:")

df=df[["sd","q25","iqr","meanfun","gender_encoded"]]
describe(shape=1,info=1)


print("\n\nFinding out correlated pairs of features:")

plt.figure(figsize=(15,10),dpi=100)
sns.heatmap(data=df.drop(["gender_encoded"],axis=1).corr(),cmap="viridis",annot=True,linewidth=0.5)
plt.show()

print("\n\nFinding less significant features out of correlated features")

sns.pairplot(df,hue="gender_encoded")
plt.show()

# exit()

print("Removing less significant correlated features:")

df=df.drop(["sd"],axis=1)
describe(shape=1,info=1)


## Saving Preprocessed Dataset

print("\n\nSaving dataset")

df.to_csv("voice_preprocessed.csv", index=False)