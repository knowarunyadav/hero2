import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('a.html')


@app.route('/titanic')
def titanic():

    return render_template('titanic.html')


@app.route('/titanic_p',methods=['POST'])
def titanic_p():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    with open('titanic.pkl', 'rb') as file:
        titanic = pickle.load(file)

    prediction = titanic.predict(final_features)
    file.close()
    output = round(prediction[0], 2)
    if output==0:
        output='will survive'
    else:
        output='will non survive'

    return render_template('titanic.html', prediction_text='{}'.format(output))


@app.route('/iris')
def iris():
    return render_template('iris.html')


@app.route('/iris_p',methods=['POST'])
def iris_p():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    with open('iris.pkl', 'rb') as file:
        titanic = pickle.load(file)


    prediction = titanic.predict(final_features)
    file.close()
    output = (prediction[0], 2)

    return render_template('iris.html', prediction_text='{}'.format(output))


@app.route('/titanic_predict_api',methods=['POST'])
def titanic_predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = titanic.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
