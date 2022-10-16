
from hashlib import algorithms_available
import numpy as np
import sklearn
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import  balanced_accuracy_score, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd
import pdb
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from mlrose_hiive.algorithms import ExpDecay
from mlrose_hiive import algorithms
from mlrose_hiive.neural import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import time


def get_data(filename):
    data = pd.read_csv(filename)
    return data



def runNNOpt():
    fname = "Data/water_potability.csv"
    data = get_data(fname)
    data = data.dropna()
    X =  data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    modelRHC = None
    modelSA = None
    modelGA = None
    modelMIMIC = None

    rhc_params = {}
    sa_params = {}
    ga_params = {}
    mimic_params = {}

    rhc_params["max_iter"] = 10000
    rhc_params["max_attempts"] = 10000
    rhc_params["max_restarts"] = 200

    sa_params["max_iter"] = 5000
    sa_params["max_attempts"] = 1000
    sa_params["decay_rate"] = 0.9 
    sa_params["min_temp"] = .001
    sa_params["init_temp"] = 1

    ga_params["max_iter"] = 500
    ga_params["max_attempts"] = 100
    ga_params["pop_size"] = 300
    ga_params["mutation_prob"] = 0.1

    
    modelBP = Pipeline([
                ('normalizer', StandardScaler()),
                ('clf', MLPClassifier(hidden_layer_sizes=10, alpha=0.0003 , random_state=0, max_iter=1000))
                ])

    modelRHC = NeuralNetwork(hidden_nodes=[10],
                            activation="relu",
                            algorithm="random_hill_climb",
                            max_iters=rhc_params["max_iter"],
                            max_attempts=rhc_params["max_attempts"],
                            bias=True,
                            learning_rate=0.0003,
                            is_classifier=True,
                            early_stopping=False,
                            random_state=1,
                            curve=True)

    decay_schdule = algorithms.ExpDecay(init_temp=sa_params["init_temp"], min_temp=sa_params["min_temp"], exp_const=sa_params["decay_rate"])
    modelSA = NeuralNetwork(hidden_nodes=[10],
                            activation="relu",
                            algorithm="simulated_annealing",
                            max_iters=sa_params["max_iter"],
                            max_attempts=sa_params["max_attempts"],
                            bias=True,
                            learning_rate=0.01,
                            schedule=decay_schdule,
                            is_classifier=True,
                            early_stopping=False,
                            random_state=1,
                            curve=True)

    modelGA = NeuralNetwork(hidden_nodes=[10],
                            activation="relu",
                            algorithm="genetic_alg",
                            max_iters=ga_params["max_iter"],
                            max_attempts=ga_params["max_attempts"],
                            pop_size=ga_params["pop_size"],
                            mutation_prob=ga_params["mutation_prob"],
                            bias=True,
                            learning_rate=0.01,
                            is_classifier=True,
                            early_stopping=False,
                            random_state=1,
                            curve=True)

    # modelSGD = NeuralNetwork(hidden_nodes=[10],
    #                         activation="relu",
    #                         algorithm="gradient_descent",
    #                         max_iters=rhc_params["max_iter"],
    #                         bias=True,
    #                         learning_rate=0.01,
    #                         is_classifier=True,
    #                         early_stopping=False,
    #                         random_state=1,
    #                         curve=False)

    scores = []
    train_times = []
    query_times = []
    iters = [0]

    start = time.time()
    modelBP.fit(x_train,y_train)
    train_times.append(time.time()-start)

    start = time.time()
    modelRHC.fit(x_train,y_train)
    iters.append(modelRHC.fitness_curve.shape[0])
    train_times.append(time.time()-start)

    start = time.time()
    modelSA.fit(x_train,y_train)
    iters.append(modelSA.fitness_curve.shape[0])
    train_times.append(time.time()-start)

    start = time.time()
    modelGA.fit(x_train,y_train)
    iters.append(modelGA.fitness_curve.shape[0])
    train_times.append(time.time()-start)

    start = time.time()
    y_pred_bp = modelBP.predict(x_test)
    query_times.append(time.time()-start)

    start = time.time()
    y_pred_rhc = modelRHC.predict(x_test)
    query_times.append(time.time()-start)

    start = time.time()
    y_pred_sa = modelSA.predict(x_test)
    query_times.append(time.time()-start)

    start = time.time()
    y_pred_ga = modelGA.predict(x_test)
    query_times.append(time.time()-start)

    bp_score = balanced_accuracy_score(y_pred_bp,y_test)
    rhc_score = accuracy_score(y_pred_rhc,y_test)
    sa_score = balanced_accuracy_score(y_pred_sa,y_test)
    ga_score = balanced_accuracy_score(y_pred_ga,y_test)

    scores = [bp_score, rhc_score, sa_score, ga_score]

    data_df = pd.DataFrame()
    data_df["scores"] = scores
    data_df["train_time"] = train_times
    data_df["query_time"] = query_times
    data_df["Iters"] = iters

    data_df.to_csv('./results/{}.csv'.format('NeuralNet'))
    print("Saved aggregate for Neural Networks \n")





    

        