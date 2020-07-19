import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.metrics import accuracy_score
app = Flask(__name__)
model = pickle.load(open('dt_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    current_loan_amount = request.form['current_loan_amount']
    loan_term = request.form['loan_term']
    credit_score = request.form['credit_score']
    annual_income = request.form['annual_income']
    years_in_industry = request.form['years_in_industry']
    past_credit_problems = request.form['past_credit_problems']
    had_bankruptcy = request.form['had_bankruptcy']
    new_record = [[current_loan_amount, loan_term, credit_score, annual_income, years_in_industry, past_credit_problems, had_bankruptcy]]
    prediction = model.predict(new_record)
    if prediction == 1:
        result = 'Accepted'
    else:
        result = 'Declined'
    return render_template('index.html', prediction_text='Your loan is going to be: {}'.format(result))  #  rendering the predicted result


if __name__ == "__main__":
    app.run(debug=True)