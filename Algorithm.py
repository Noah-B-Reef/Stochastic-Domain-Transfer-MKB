# import required libraries
import ray
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
import math
import pandas as pd
import random
from numpy.linalg import pinv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_kernels
import time
r = random.Random(500)
np_config.enable_numpy_behavior()
tf.debugging.set_log_device_placement(False)
ray.shutdown()

# Kernel Matricies

# Generate Kernel Matrix for Linear Kernel
def linMat():
  return lambda X : pairwise_kernels(X, metric="linear")

# Generate Kernel Matrix for Sigmoid Kernel
def sigmoidMat(gamma):
  return lambda X : pairwise_kernels(X, metric="sigmoid", gamma=gamma)

# Generate Kernel Matrix for Polynomial Kernel
def polyMat(deg):
  return lambda X : pairwise_kernels(X, metric="poly", degree=deg)

# Generate Kernel Matrix for RBF Kernel
def rbfMat(gamma):
  return lambda X : pairwise_kernels(X, metric="rbf", gamma=gamma)

# Generate Kernel Matrix for Laplace Kernel
def laplaceMat(gamma):
  return lambda X : pairwise_kernels(X , metric="laplacian", gamma=gamma)

# Set of Base Kernels for Creating Kernel Matrix K

kernels = [linMat(), sigmoidMat(2**-6), sigmoidMat(2**-3), sigmoidMat(1), sigmoidMat(2**3),
           sigmoidMat(2**6), polyMat(1), polyMat(2), polyMat(3), polyMat(4),
           rbfMat(2**-6), rbfMat(2**-3), rbfMat(1), rbfMat(2**3), rbfMat(2**6),
           laplaceMat(2**-6), laplaceMat(2**-3), laplaceMat(1), laplaceMat(2**3), laplaceMat(2**6)]


# kernels = [linMat(), sigmoidMat(2**-3), polyMat(4), rbfMat(2**-3)]

# Get Subset of Kernels for Trial
def getKernels(kernels, numOfKernels, weights=[1 for i in range(len(kernels))]):
  """
  Selects kernels from list of base kernels for a given boosting trial of SDTMKB

  Args:
      kernels: List of base kernels
      numOfKernels: Number of kernels to be selected from 'kernels' list
      weights: List of probabilities for a given base kernel to be selected

  Returns:
      returns 'numOfKernels' many kernels from 'kernels' using distribution provided by 'weights'
  """
  return r.choices(kernels, weights=weights, k=numOfKernels)

def computeK(d, kernelMats):
  sum = 0

  for i in range(len(d)):
    sum += d[i] * kernelMats[i]

  return sum

def computeKernel(d, kernels, x_i, x):

  x = [list(x_i)] + [list(x)]

  sum = 0

  for i in range(len(d)):
    sum += d[i] * kernels[i](x)[0][1]

  return sum


# Compute Weight Matrix S
def computeS(numOfAuxiliary, numOfTarget):
  s = np.array([1/numOfAuxiliary for i in range(numOfAuxiliary)]
               + [-1/numOfTarget for i in range(numOfTarget)]).reshape(numOfAuxiliary + numOfTarget, 1)
  return np.matmul(s, s.T)


# Compute vector P
def computeP(K,S):
  p = []
  for k in K:
    p.append(np.trace(np.matmul(k,S)))
  return np.array([p])

#  Implementation of learning dual coefficients alpha
def computeAlpha(K, y_train):

  # Make SVM classifer using precomputed K = sum(d_m * K_m)
  clf = SVC(kernel="precomputed")
  clf.fit(K, y_train)

  # Array of Support Vectors of SVM
  SV = clf.support_

  # Dual Coefficients (Alpha) of Support Vectors [0 for non-support vectors]
  dual_coef = clf.dual_coef_
  alpha = [0] * len(y_train)
  y = []
  for i in range(dual_coef.shape[1]):
        temp = dual_coef[0][i]
        if temp < 0:
            temp = 0 - temp
        alpha[SV[i]] = temp

  return np.array(alpha), SV

# Reduced Gradient Descent
def computeDirection(alpha, kernelMats, y, epsilon, p, d, theta):
  y_mat = tf.tensordot(y,y,axes=0)
  grad_J = compute_dJ(kernelMats, y_mat, alpha)
  g = np.array(d) + theta * tf.matmul(pinv((tf.matmul(p.T, p) + epsilon * np.eye(len(kernelMats)))),grad_J.reshape(len(kernelMats),1)).T
  return np.array(g[0])

# Implements DTMKL algorithm
def DTMKL(kernels, X_train, y_train, C, numOfAuxiliary, numOfTarget):
  """
  Creates a Binary Classifier using the DTMKL algorithm

  Args:
      kernels: List of base kernels to create classifier
      X_train: Feature vectors of dataset
      y_train: labels of vectors in dataset
      numOfAuxiliary: Number of samples from the Auxiliary Domain
      numOfTarget: Number of samples from the Target Domain

  Returns:
      Returns dual coefficients 'alpha', kernel weights 'd', and list of support vectors 'SV'
  """


  # Establish Necessary Parameters
  MAX_ITER = 10
  epsilon = 10**(-2)
  theta = C
  step_size = (math.sqrt(5) - 1) / 2
  eta = step_size

  # Create Kerenel Matrices
  kernelMats = [kern(X_train) for kern in kernels]

  # Create Matrix P
  p = computeP(kernelMats, computeS(numOfAuxiliary,numOfTarget))

  # intialize weights d
  d = [1/len(kernels)] * len(kernels)

  # implement loop
  for i in range(MAX_ITER):

    # Combined Kernel Matrix K
    start = time.time()
    K = computeK(d, kernelMats)
    end = time.time()


    # compute Alpha
    start = time.time()
    alpha, SV = computeAlpha(K, y_train)
    end = time.time()


    # compute d
    d = d - eta * computeDirection(alpha, kernelMats, y_train, epsilon, p, d, theta)

    # Project d onto the feasible set M
    for i in range(len(d)):
      if d[i] < 0:
        d[i] = 0

    d_new = [num/sum(d) for num in d]
    d = d_new

    eta = eta * step_size

  return alpha, d, SV

def sign(num):
  if num < 0:
    return -1
  else:
    return 1

def getSamples(sizeOfData, weights, dataset):


  samples = dataset.copy(deep=True)
  samples = pd.concat([samples.sample(sizeOfData, weights=weights), samples.groupby("label", group_keys=False).apply(lambda df: df.sample(1))])

  numOfAuxiliary = (samples['Domain'] == -1).sum()
  numOfTarget = (samples['Domain'] == 1).sum()
  samples.drop(["Domain"], inplace=True, axis=1)

  # Split feature vector from Label for Training Data
  y_samples = samples["label"].values
  samples.drop(["label"], inplace=True, axis=1)
  X_samples = samples.to_numpy()

  return X_samples, y_samples, numOfAuxiliary, numOfTarget


def computeLoss(clf, X_train, y_train):
  """
  Computes the error of the classifier over a given dataset

  Args:
      clf: Lambda function that represents the classifier
      X_train: 2D-numpy array that represents the matrix of feature vectors of the dataset
      y_train: 1D-numpy array that represents the labels of each feature vector  of the dataset

  Returns:
      Returns the error of the classifier over the dataset
  """

  # Parallelizes Error Computation
  '''
  ray.init(log_to_driver=False)
  X_split = np.array_split(X_train, len(X_train) * 0.20)
  y_split = np.array_split(y_train, len(X_train) * 0.20)
  '''

  total_corr = 0
  for i in range(len(X_train)):
      if clf(X_train[i]) == y_train[i]:
          total_corr += 1

  # total_corr += np.sum(ray.get([Error.remote(clf, X_split[i], y_split[i]) for i in range(len(X_split))]))

  # ray.shutdown()
  return 1 - total_corr/len(y_train)

'''
@ray.remote
def Error(clf, X_split, y_split):
  sum = 0
  for i in range(len(X_split)):
    if (clf(X_split[i]) == y_split[i]):
      sum += 1
  return sum
'''

def createClassifier(kernels, X_train, y_train, C, numOfAuxiliary, numOfTarget):
  alpha, d, SV = DTMKL(kernels, X_train, y_train, C, numOfAuxiliary, numOfTarget)
  return lambda X : sign(sum([alpha[i] * y_train[i] * computeKernel(d, kernels, X_train[i], X) for i in SV]))

# this function was implemented in: https://github.com/gjtrowbridge/simple-mkl-python/blob/master/helpers.py
def compute_dJ(kernel_matrices, y_mat, alpha):
    #this function computes the gradient given alphas derived from a SVM solution
    M = len(kernel_matrices)#kernel_matrices will be M-by-N-by-N 3D matrices, M kernels, N samples
    dJ = np.zeros(M)#gradient container for a given d vector
    for m in range(M):
        kernel_matrix = kernel_matrices[m]#pick one kernel matrix out of M kernel matrices
        #dJ[m] contains the gradient for each dm
        #Equation (11): -0.5*sum(alpha_i*alpha_j*y_i*y_j*K_m(xi,xj)),for all i,j
        dJ[m] = -0.5 * alpha.dot(np.multiply(kernel_matrix,y_mat)).dot(alpha.T)#based on equation (11) in the simpleMKL paper, it is correct
    return dJ

def stochasticDTMKB(T, kernels, numOfKernels, sizeOfData, dataset,stochastic,samplingDecay=2**(-5)):

  """
  Creates Classifier Created Using the Stochastic Domain Transfer Multiple Kernel Boosting Algorithm

  Args:
      T: Number of Boosting Trials
      kernels: List of base kernels for the algorithm to choose from during each boosting trial
      numOfKernels: Number of kernels chosen from the list of base kernels to used in training weak classifer for a given boosting trial
      sizeOfData: Number of samples from the training dataset to be used for traiing weak classifier for a given boosting trial
      stochastic: Flag for using stochastic implementation of the algorithm
      samplingDecay: Controls the rate of updating probabilities for base kernels, default=2**(-5)

  Returns:
      Lambda function that takes a numpy-array as input and returns the classifiers prediction of its label {-1,1}
  """

  d = dataset.copy(deep=True)
  d.drop(['Domain'], inplace=True, axis=1)

  # Split feature vector from Label for Training Data
  y_train = d["label"].values
  d.drop(["label"], inplace=True, axis=1)
  X_train = d.to_numpy()

  # Initializes weights for Data and Kernel Distributions
  D_weights = [1/len(dataset) for i in range(len(dataset))]
  S_weights = [1 for i in range(len(kernels))]

  # Weak Classifiers and their loss
  h_t = []
  loss = []
  alpha_t = []

  # Adaboosting
  for t in range(T):
    print("t = " + str(t))
    # Get sample data
    X_samples, y_samples, numOfAuxiliary, numOfTarget = getSamples(sizeOfData, D_weights, dataset)


    # Get Kernels
    if stochastic == True:
      sampleKernels = getKernels(kernels, numOfKernels, S_weights)

    else:
      sampleKernels = kernels

    # Train Weak Classifer
    f = createClassifier(sampleKernels, X_samples, y_samples, 1, numOfAuxiliary, numOfTarget)



    # Compute error of weak classifier over full dataset
    start = time.time()
    error = float(computeLoss(f, X_train, y_train))
    end = time.time()


    kernel_indicies = [kernels.index(kern) for kern in sampleKernels]
    if stochastic:
        # Update S_{t+1}
        for i in range(len(kernel_indicies)):
          S_weights[kernel_indicies[i]] = S_weights[kernel_indicies[i]]*(samplingDecay**error)

        # Normalize S_{t+1}
        S_weights = [weight/max(S_weights) for weight in S_weights]

    # Calculate weight of weak classifier, alpha
    if error == 0:
        alpha = 0.5
        h_t.append(f)
        alpha_t.append(alpha)
    # else:
    elif error < 0.5:
        alpha = 0.5 * np.log((1-error)/error)
        h_t.append(f)
        alpha_t.append(alpha)


    else:
       alpha = 0


    # Update distribution D_t
    for i in range(len(X_train)):

      if f(X_train[i]) != y_train[i]:
        D_weights[i] = D_weights[i]/max(D_weights) * np.exp(-alpha)

      else:
        D_weights[i] = D_weights[i]/max(D_weights) * np.exp(alpha)

  return lambda X: sign(np.sum([alpha_t[i] * h_t[i](X) for i in range(len(alpha_t))]))



