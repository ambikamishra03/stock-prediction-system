# -*- coding: utf-8 -*-
"""
    Miscellaneous Functions for Regression File (Updated for Python 3 & sklearn >=1.0)
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor, BaggingRegressor, AdaBoostRegressor,
    GradientBoostingRegressor, RandomForestClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import os
from Neural_Network import NeuralNet

# ---------------------- DATA LOADING ----------------------

def load_dataset(path_directory, symbol):
    """
    Import DataFrame from Dataset.
    """
    path = os.path.join(path_directory, symbol)
    out = pd.read_csv(path, index_col=2, parse_dates=[2])
    out.drop(out.columns[0], axis=1, inplace=True)
    return [out]


def count_missing(dataframe):
    """Count number of NaN in dataframe"""
    return dataframe.isna().sum().sum()


def addFeatures(dataframe, adjclose, returns, n):
    """
    Adds moving average and percentage change features.
    """
    return_n = adjclose[9:] + f"Time{n}"
    dataframe[return_n] = dataframe[adjclose].pct_change(n)

    roll_n = returns[7:] + f"RolMean{n}"
    dataframe[roll_n] = dataframe[returns].rolling(window=n).mean()

    exp_ma = returns[7:] + f"ExponentMovingAvg{n}"
    dataframe[exp_ma] = dataframe[returns].ewm(halflife=n).mean()


def mergeDataframes(datasets):
    """Merge Datasets into a single DataFrame"""
    return pd.concat(datasets)


def applyTimeLag(dataset, lags, delta):
    """
    Apply time lag to dataset columns.
    """
    maxLag = max(lags)
    columns = dataset.columns[::(2 * max(delta) - 1)]
    for column in columns:
        newcolumn = f"{column}{maxLag}"
        dataset[newcolumn] = dataset[column].shift(maxLag)
    return dataset.iloc[maxLag:-1, :]


# ---------------------- CLASSIFICATION ----------------------

def prepareDataForClassification(dataset, start_test):
    """Prepare labeled dataset for classification"""
    le = preprocessing.LabelEncoder()

    dataset['UpDown'] = np.where(dataset['Return_Out'] >= 0, 'Up', 'Down')
    dataset['UpDown'] = le.fit_transform(dataset['UpDown'])

    features = dataset.columns[1:-1]
    X = dataset[features]
    y = dataset['UpDown']

    X_train = X[X.index < start_test]
    y_train = y[y.index < start_test]

    X_test = X[X.index >= start_test]
    y_test = y[y.index >= start_test]

    return X_train, y_train, X_test, y_test


def prepareDataForModelSelection(X_train, y_train, start_validation):
    """Split data into train and validation sets"""
    X = X_train[X_train.index < start_validation]
    y = y_train[y_train.index < start_validation]

    X_val = X_train[X_train.index >= start_validation]
    y_val = y_train[y_train.index >= start_validation]

    return X, y, X_val, y_val


def performClassification(X_train, y_train, X_test, y_test, method, parameters=None):
    """Perform Classification using multiple classifiers"""
    if parameters is None:
        parameters = {}

    print(f'Performing {method} Classification...')
    print(f'Size of train set: {X_train.shape}, test set: {X_test.shape}')

    classifiers = [
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        KNeighborsClassifier(),
        SVC(degree=3, C=10, epsilon=.01, kernel='rbf'),
        AdaBoostClassifier(**parameters),
        GradientBoostingClassifier(n_estimators=100),
        QDA(),
    ]

    scores = []
    for clf in classifiers:
        score = benchmark_classifier(clf, X_train, y_train, X_test, y_test)
        scores.append(score)

    print(scores)


def benchmark_classifier(clf, X_train, y_train, X_test, y_test):
    """Train and evaluate classifier"""
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return accuracy


# ---------------------- REGRESSION ----------------------

def getFeatures(X_train, y_train, X_test, num_features=5):
    """Feature selection"""
    ch2 = SelectKBest(chi2, k=num_features)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    return X_train, X_test


def performRegression(dataset, split, symbol, output_dir):
    """Perform Regression using multiple models"""
    features = dataset.columns[1:]
    index = int(np.floor(dataset.shape[0] * split))
    train, test = dataset[:index], dataset[index:]
    print(f'Size of train set: {train.shape}, test set: {test.shape}')

    out_params = (symbol, output_dir)
    output = dataset.columns[0]
    predicted_values = []

    regressors = [
        RandomForestRegressor(n_estimators=10, n_jobs=-1),
        SVR(C=1000, kernel='rbf', epsilon=0.1, gamma=1),
        BaggingRegressor(),
        AdaBoostRegressor(),
        KNeighborsRegressor(),
        GradientBoostingRegressor(),
    ]

    for reg in regressors:
        predicted_values.append(
            benchmark_model(reg, train, test, features, output, out_params)
        )

    # Neural Network model
    classifier = NeuralNet(50, learn_rate=1e-2)
    predicted_values.append(
        benchmark_model(
            classifier, train, test, features, output, out_params,
            fine_tune=False, maxiter=1000, SGD=True, batch=150, rho=0.9
        )
    )

    print('-' * 80)

    mse_scores = []
    r2_scores = []

    for pred in predicted_values:
        mse_scores.append(mean_squared_error(test[output].values, pred))
        r2_scores.append(r2_score(test[output].values, pred))

    print(mse_scores, r2_scores)
    return mse_scores, r2_scores


def benchmark_model(model, train, test, features, output, output_params, *args, **kwargs):
    """Train model, predict, and plot results"""
    print('-' * 80)
    model_name = str(model).split('(')[0].replace('Regressor', ' Regressor')
    print(model_name)

    symbol, output_dir = output_params

    model.fit(train[features].values, train[output].values, *args, **kwargs)
    predicted_value = model.predict(test[features].values)

    plt.figure(figsize=(8, 4))
    plt.plot(test[output].values, color='g', label='Actual Value')
    plt.plot(predicted_value, color='b', linestyle='--', label='Predicted Value')
    plt.xlabel('Samples')
    plt.ylabel('Output Value')
    plt.title(model_name)
    plt.legend(loc='best')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{symbol}_{model_name}.png"), dpi=120)
    plt.close()

    return predicted_value
