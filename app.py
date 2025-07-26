from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
similarity_model = joblib.load('similarity.pkl')
medicine_model = joblib.load('medicine_list.pkl')

# Load dataset (for medicine name lookup)
medicine_df = pd.read_csv('medicine.csv')  # Make sure the file is in the root directory

@app.route('/')
def home():
    medicine_names = list(medicine_df['Drug_Name'])
    return render_template('index.html', medicines=medicine_names)

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        selected_medicine = request.form['medicine']

        # Find index of selected medicine
        idx = medicine_df[medicine_df['Drug_Name'] == selected_medicine].index[0]

        # Get similarity scores and top 5 similar
        sim_scores = list(enumerate(similarity_model[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        recommended = [medicine_df.iloc[i[0]].Drug_Name for i in sim_scores]

        return render_template('result.html', selected=selected_medicine, recommendations=recommended)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)