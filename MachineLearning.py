#import stuff
from pathlib import Path
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# load metadata
print(Path.cwd()) 
metadata = pd.read_csv(r'./information.csv') 
#^^change directory if needed^^


metadata.head()

# function of feature extraction MFCCs
def feature_extractor(file):
    y, sr = librosa.load(file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_scaled = np.mean(mfccs.T, axis=0)

    return mfcc_scaled

# load all of dataset
preprocessed = []
for i in metadata.iterrows():
    file_name = os.path.join(i[1]["directory"]+i[1]["fname"])
    label = i[1]["label"]
    mfccs = feature_extractor(file_name)
    preprocessed.append([mfccs, label])

# make DataFrame
extracted_features_df=pd.DataFrame(preprocessed,columns=['feature','class'])
extracted_features_df=pd.DataFrame(extracted_features_df['feature'].values.tolist()).add_prefix('feature').join(extracted_features_df)
extracted_features_df=extracted_features_df.drop('feature', axis=1)
extracted_features_df.head(10)

### TRAINING MODELS ###
# split train data and test data
data_x = extracted_features_df.drop(['class'], axis=1)
data_y = extracted_features_df['class']

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)


### SVM CLASSIFIERS ###
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
pred_y = svc.predict(X_test)

plot = confusion_matrix(y_test, pred_y)

#Setting the attributes
fig, px = plt.subplots(figsize=(7.5, 7.5))
px.matshow(plot, cmap=plt.cm.YlOrRd, alpha=0.5)
for m in range(plot.shape[0]):
    for n in range(plot.shape[1]):
        px.text(x=m,y=n,s=plot[m, n], va='center', ha='center', size='xx-large')

# Sets the labels
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Actuals', fontsize=16)
plt.title('SVM', fontsize=15)
plt.show() #need to close out of the matrix windows to continue to run

### RANDOM FOREST###
randomforest = RandomForestClassifier()
randomforest.fit(X_train, y_train)

pred_y = randomforest.predict(X_test)

plot = confusion_matrix(y_test, pred_y)

fig, px = plt.subplots(figsize=(7.5, 7.5))
px.matshow(plot, cmap=plt.cm.YlOrRd, alpha=0.5)
for m in range(plot.shape[0]):
    for n in range(plot.shape[1]):
        px.text(x=m,y=n,s=plot[m, n], va='center', ha='center', size='xx-large')

# Sets the labels
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Actuals', fontsize=16)
plt.title('Random Forest', fontsize=15)
plt.show() #need to close out of the matrix windows to continue to run

###LGBM###
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)

pred_y = lgbm.predict(X_test)

plot = confusion_matrix(y_test, pred_y)

fig, px = plt.subplots(figsize=(7.5, 7.5))
px.matshow(plot, cmap=plt.cm.YlOrRd, alpha=0.5)
for m in range(plot.shape[0]):
    for n in range(plot.shape[1]):
        px.text(x=m,y=n,s=plot[m, n], va='center', ha='center', size='xx-large')

# Sets the labels
plt.xlabel('Predictions', fontsize=16)
plt.ylabel('Actuals', fontsize=16)
plt.title('LGBM', fontsize=15)
plt.show() #need to close out of the matrix windows to continue to run

print('done')
