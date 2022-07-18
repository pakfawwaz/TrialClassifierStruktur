from flask import Flask,render_template,url_for,request

import pickle
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df= pd.read_excel("DatasetNLPmama.xlsx")
	# Features and Labels
    df['Struktur'] = df['Struktur'].map({'SPOK': 0, 'SPO': 1})
    X = df['Kalimat']
    y = df['Struktur']
	
	# Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X) # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier

    clf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)