import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def plot_corr(df, size=11):
    corr = df.corr()    # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)   # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
    print_data(corr, 'Correlation')
    #plt.show()

def map_boolean_to_integer():
    return {True : 1, False: 0}

def check_boolean_ration(df):
    num_true = len(df.loc[df['diabetes'] == 1])
    num_false = len(df.loc[df['diabetes'] == 0])
    print('=========> Cases ration')
    print("Number of True cases:  {0} ({1:2.2f}%)".format(num_true, (num_true/ (num_true + num_false)) * 100))
    print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (num_false/ (num_true + num_false)) * 100))
    print('=========')


def verify_split_data(df, X_train, X_test, y_train, y_test):
    print(f'\n=========> Split Data')
    print("{0:0.2f}% in training set".format((len(X_train)/len(df.index)) * 100))
    print("{0:0.2f}% in test set".format((len(X_test)/len(df.index)) * 100))

    # Verifying predicted values was split correctly
    print("Original True  : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 1]), (len(df.loc[df['diabetes'] == 1])/len(df.index)) * 100.0))
    print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 0]), (len(df.loc[df['diabetes'] == 0])/len(df.index)) * 100.0))
    print("")
    print("Training True  : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train) * 100.0)))
    print("Training False : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train) * 100.0)))
    print("")
    print("Test True      : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test) * 100.0)))
    print("Test False     : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test) * 100.0)))
    print('=========\n')

def show_metrics(y, y_predicted, model_name=''):
    print(f'\n=========> Metrics ({model_name}):')
    print("Confusion Matrix (False=0 upper left and 1=True lower right)")
    print("{0}".format(metrics.confusion_matrix(y, y_predicted)))
    print("")


    print("Classification Report (False=0 on first line. 1=true on second line)")
    print("")
    print(metrics.classification_report(y, y_predicted))
    print('=========\n')

def show_accuracy(y, y_predicted, data_set_name='Training', model_name='Model'):
    accuracy = metrics.accuracy_score(y, y_predicted)
    print_data(data=round(accuracy, 4), name=f'{model_name} Accuracy on {data_set_name} data')

def print_data(data, name=''):
    print(f'\n=========> {name}')
    print(data)
    print('=========\n')