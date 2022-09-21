from flask import Flask, request
# import pickle
# import pandas as pd
# import numpy as np

app = Flask(__name__)

#
# base_path = r"C:\Users\mfuser\Desktop\ITC\DevOPs\Flask"
# model_file_name = "churn_model.pkl"
# X_test_file_name = "X_test.csv"
# y_pred_file_name = "preds.csv"


@app.route('/help')
def help():
    msg = 'features and values for this model:<br>'
    params = {'is_male': {0,1}, 'num_inters': '0 - 16', 'late_on_payment': {0, 1}, 'age': 'some_age', 'years_in_contract': '0 - 7+'}
    return '<br>'.join([msg]+dict_to_str(params))


@app.route('/hi')
def hi():
    return 'hi'

#
@app.route('/predict')
def predict():
    pass
    # pd.read_csv('water.csv')
#     args = request.args
#     # X_input = args.get('predict_churn')
#     # create a pandas dataframe from the input
#     X_input = {k: float(v) for k,v in args.items()}
#     X = pd.DataFrame.from_dict(X_input, orient='index').T
#
#     # return str(X.iloc[0,:].values)
#
#     # cols expected
#     cols_model = pd.Index(clf.feature_names_in_)
#
#     # add NA for cols missing
#     X[cols_model.difference(X.columns)] = np.nan
#
#     # reorder X according to trained model
#     X = X[cols_model]
#
#     # predict
#     y_pred = clf.predict(X)
#
#     if y_pred:
#         return "Gonna churn"
#     else:
#         return "Not gonna churn"


if __name__ == '__main__':
    # clf = get_model()
    app.run(host="0.0.0.0", port=80)