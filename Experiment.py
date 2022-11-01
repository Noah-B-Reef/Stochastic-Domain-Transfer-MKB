# import required libraries
#from tkinter import W
# import ray
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
import math
import seaborn as sns
import pandas as pd
import random
from numpy.linalg import pinv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_kernels
import time
from Algorithm import *
r = random.Random(500)
np_config.enable_numpy_behavior()
tf.debugging.set_log_device_placement(False)


# Experiment Implementation

train = pd.read_csv(r'/data03/home/nbreef/Stochastic-Domain-Transfer-MKB/Experiment Data/Data/scaled_adj_train.csv')
set_test = pd.read_csv(r'/data03/home/nbreef/Stochastic-Domain-Transfer-MKB/Experiment Data/Data/scaled_adj_test.csv')

# Split feature vector from Label for Testing Data
set_test.drop("Domain", inplace=True, axis=1)
y_test = set_test["label"].values
set_test.drop("label", inplace=True, axis=1)
X_test = set_test.to_numpy()
df = pd.DataFrame(columns = ['T', 'M', 'TP', 'TN', 'FP', 'FN','Accuracy'])
#df = pd.read_csv('/data03/home/nbreef/Stochastic-Domain-Transfer-MKB/Experiment Data/ExpData_T_small.csv')

for T in [1]:
  for k in [6]:
    print('running')
    clf = stochasticDTMKB(T, kernels, k, int(len(train)*0.2), train, True)


    TP = 0
    TN = 0
    FP = 0
    FN = 0
    corr = 0

    for i in range(len(X_test)):
      if clf(X_test[i]) == y_test[i] and y_test[i] == 1:
        TP += 1
        corr +=1

      elif clf(X_test[i]) == y_test[i] and y_test[i] == -1:
        TN += 1
        corr += 1

      elif clf(X_test[i]) != y_test[i] and clf(X_test[i]) == 1:
        FP += 1

      else:
        FN += 1
    df1 = pd.DataFrame([[T, k, TP, TN, FP, FN, corr/len(X_test)]], columns=['T', 'M', 'TP', 'TN', 'FP', 'FN','Accuracy'])
    df = pd.concat([df, df1], ignore_index=True)
    df.to_csv("/data03/home/nbreef/Stochastic-Domain-Transfer-MKB/Experiment Data/ExpData_Test", index=False)
