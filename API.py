from flask import Flask, request
from arnona_inference import predict_arnona
from electricity_inference import predict_electricity
from water_inference import predict_water
import pandas as pd
import os


# os.chdir(r'C:\Users\mfuser\jupyter_notebook\Side Pros\Hackaton\GIT')

app = Flask(__name__)


@app.route('/help')
def help():
    return 'helpppp'


@app.route('/infer_arnona')
def infer_arnona():
    args = request.args

    X_input = {k: v for k, v in args.items()}

    if 'month' in X_input:
        del X_input['month']

    X = pd.DataFrame.from_dict(X_input, orient='index').T

    res = predict_arnona(X)

    return str(res[0])


@app.route('/infer_electricity')
def infer_electricity():
    args = request.args

    X_input = {k: v for k, v in args.items()}
    X = pd.DataFrame.from_dict(X_input, orient='index').T

    res = predict_electricity(X)

    return str(res[0])


@app.route('/infer_water')
def infer_water():
    args = request.args

    X_input = {k: v for k, v in args.items()}
    X = pd.DataFrame.from_dict(X_input, orient='index').T

    res = predict_water(X)

    return str(res[0])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)