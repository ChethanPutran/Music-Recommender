import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

music_data = pd.read_csv('music.csv')
# print(music_data)

#Creates new dataset without column 'genre'
X=music_data.drop(columns=['genre'])
y=music_data['genre']

#Splitting dataset for training and testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Building model
model = DecisionTreeClassifier()

#Inputting training dataset to the model
model.fit(X_train,y_train)

"""
Storing the model into joblib file
joblib.dump(model,'music-recommender.joblib')

Calling trained model to make predictions
model = joblib.dump(model,'music-recommender.joblib')"""


#Making predictions
predictions=model.predict(X_test)

#Evaluating the accuracy
score = accuracy_score(y_test,predictions)

print(score)