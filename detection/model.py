#! /usr/bin/env python

# =================================
# Modules
# =================================

import os
import glob
import numpy as np
import scipy
import chainer.functions as F
from scipy import io
from scipy.io import wavfile
from scipy.io.wavfile import read as wavread
from scikits.talkbox.features import mfcc
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from chainer import cuda, Chain, Variable, FunctionSet, optimizers

from constant import *

# =================================
# Global Variables
# =================================

MODEL_LR = LogisticRegression
MODEL_SVM = LinearSVC
MODEL_NN = lambda: Chain(
              l1=F.Linear(13, 256),
              l2=F.Linear(256, 256),
              l3=F.Linear(256, 3)
            )
N_EPOCH = 100
BATCHSIZE = 20

# =================================
# Functions
# =================================

def write_ceps(ceps, path):
  output_path = os.path.splitext(path)[0] + ".ceps"
  np.save(output_path, ceps)

def read_as_mfcc(path):
  sample_rate, X = wavread(path)
  ceps, mspec, spec = mfcc(X)
  return ceps

def create_ceps(path):
  sample_rate, X = wavread(path)
  ceps, mspec, spec = mfcc(X)
  write_ceps(ceps, path)

def trim_ceps(ceps, ratio=0.1):
  count = len(ceps)
  start = int(count*ratio)
  end = int(count*(1-ratio))
  return ceps[start:end]

def read_ceps(labels, directory=DATA_DIR):
  X, y = [], []
  for label in labels:
    for path in glob.glob(os.path.join(directory, label, "*.ceps.npy")):
      ceps = np.load(path)
      X.append(np.mean(trim_ceps(ceps), axis=0))
      y.append(label)
  return np.array(X), np.array(y)

def create_ceps_all(directory=DATA_DIR):
  for path in glob.glob(os.path.join(directory, "*", "*_part_*.wav")):
    create_ceps(path)

# def logistic_regression_model(classifier, X):
#   a0 = classifier.intercept_
#   a1 = classifier.coef_
#   denominator = 1 + np.exp(a0+a1*X)
#   return 1 / denominator

def error_count(xs, ys):
  return np.sum([ 1 if x == y else 0 for x, y in zip(xs, ys) ])

def validate(X, y, train_indices, test_indices, model):
    print("TRAIN:", train_indices, "TEST:", test_indices)
    classifier = model()
    train(X[train_indices], y[train_indices], classifier)
    y_preds = map(classifier.predict, [ x.reshape(1, -1) for x in X[train_indices] ])
    err = error_count(y[test_indices], y_preds)
    cm = confusion_matrix(y[test_indices], y_preds)
    return err, cm

def cross_validate(X, y, n, model):
  kf = KFold(len(X), n_folds=n, shuffle=False)
  return [ validate(X, y, train, test, model) for train, test in kf ]

def forward(x_data, y_data, model, train=True):
   x, t = Variable(x_data, volatile=not train), Variable(y_data, volatile=not train)
   h1 = F.dropout(F.relu(model.l1(x)), train=train)
   h2 = F.dropout(F.relu(model.l2(h1)), train=train)
   y = model.l3(h2)
   return F.softmax_cross_entropy(y, t), F.accuracy(y, t)

def label_index(label):
  labels = ['speech', 'silent', 'noise']
  return labels.index(label)

def train(x_train, y_train, model):
  if isinstance(model, MODEL_LR) or isinstance(model, MODEL_SVM):
    model.fit(x_train, y_train)
  else: #CNN
    N = len(x_train)
    x_train = x_train.astype(np.float32)

    label2index = np.vectorize(label_index)
    y_train = label2index(y_train).astype(np.int32)

    for epoch in xrange(1, N_EPOCH + 1):
      print 'epoch', epoch

      perm = np.random.permutation(N)
      sum_accuracy = 0
      sum_loss = 0
      for i in xrange(0, N, BATCHSIZE):
        x_batch = x_train[perm[i:i+BATCHSIZE]]
        y_batch = y_train[perm[i:i+BATCHSIZE]]

        optimizer = optimizers.Adam()
        optimizer.setup(model)

        optimizer.zero_grads()
        loss, acc = forward(x_batch, y_batch, model)
        loss.backward()
        optimizer.update()

        sum_loss     += float(loss.data) * BATCHSIZE
        sum_accuracy += float(acc.data) * BATCHSIZE

      print 'train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N)
