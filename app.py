from flask import Flask, render_template, request
import numpy as np
import pickle
filename = "titanic_decisiontree.sav"
load_model = pickle.load(open(filename, "rb"))

app = Flask(__name__)
@app.route('/')
def home():
    return render_template ('home.html')
@app.route('/pred', methods = ['POST',"GET"])
def predict():
    if request.method == "POST":
        pclass = request.form.get("pclass")
        sex = request.form.get("sex")
        sibsp = request.form.get("sibsp")
        parch = request.form.get("parch")
        fare = request.form.get("fare")
       
        input = np.array([pclass, sex, sibsp,parch,fare])
        value = input.astype(np.float_)
        pred = load_model.predict([value])[0]

        result = ''
        if pred == 1:
            result = 'Survived'
        else:
            result = 'Died'

    return render_template('predict.html', pred = '{}'.format(result))
@app.route('/plot')
def plot():
    return render_template ('plot.html')

if __name__ == '__main__':
    app.run(debug=True)