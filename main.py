from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('data.csv')

# Define a TF-IDF vectorizer to extract features from the drug contents
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the drug contents to TF-IDF feature vectors
X = tfidf.fit_transform(data['short_composition1'])

# Define a function to get similar medicines


def get_similar_medicines(medicine_name):
    medicine_idx = data[data['name'] == medicine_name].index
    if len(medicine_idx) == 0:
        return None
    cosine_similarities = cosine_similarity(X[medicine_idx], X).flatten()
    similar_medicine_indices = cosine_similarities.argsort()[::-1][1:10]
    similar_medicines = data.iloc[similar_medicine_indices][[
        'name', 'short_composition1', 'manufacturer_name', 'price', 'pack_size_label']].reset_index(drop=True)
    return similar_medicines


@app.route('/', methods=['GET', 'POST'])
def index():
    similar_medicines2 = None
    error_message = None
    if request.method == 'POST':
        medicine_name = request.form['input1']
        similar_medicines2 = get_similar_medicines(medicine_name)
        if similar_medicines2 is None:
            error_message = f"No medicine found matching '{medicine_name}'"
    return render_template('index.html', similar_medicines2=similar_medicines2, error_message=error_message)


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run('0.0.0.0', 5000)
