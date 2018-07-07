import sys
import tqdm
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error

# to ignore warnings when fitting arima
import warnings
warnings.filterwarnings("ignore")


def evaluate_arima_model(X, arima_order):
	# evaluate an ARIMA model for a given order (p,d,q)
	# source: https://machinelearningmastery.com/grid-search-arima-hyperparameters-with-python/

    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


def fit_arima(dataset):
    # define grid of parameters
    p_values = range(0, 2)
    d_values = range(0, 2)
    q_values = range(0, 2)
#     evaluate_models(series.values, p_values, d_values, q_values)
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
#                     print('ARIMA%s MSE=%.3f' % (order,mse))
                except Exception as e: 
#                     print(e)
                    continue
#     print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))                
    return best_cfg
