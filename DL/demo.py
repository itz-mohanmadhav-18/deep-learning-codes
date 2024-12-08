from flask import Flask,render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
import tensorflow

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    df = pd.read_csv('spam_ham_dataset.csv')
    x = df['text']
    y = df['label_num']
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.model_selection import train_test_split
    cv = TfidfVectorizer()
    x = cv.fit_transform(x)
    x = x.toarray()
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=99)
    model = Sequential()
    model.add(Dense(132, activation='relu'))
    model.add(Dense(132, activation='relu'))
    model.add(Dense(132, activation='relu'))
    model.add(Dense(132, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test)
    pred  = model.predict(x_test)
    for i in range(len(pred)):
        if pred[i] > 0.5:
            pred[i] = 1
        else:
            pred[i] = 0
    print(pred[0])

if __name__== "__main__":
    app.run()