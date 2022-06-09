from enum import unique
import enum
from lib2to3.pytree import Base
from msilib.schema import Class
import scipy
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn import datasets, random_projection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy.random import RandomState
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_X_y, check_array
from scipy.spatial import distance
from scipy import stats
from scipy.spatial import distance
from sklearn.model_selection import RepeatedStratifiedKFold
from tabulate import tabulate
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector, SelectPercentile, SelectKBest
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import rankdata
from scipy.stats import ranksums
from scipy.stats import ttest_rel, ttest_ind

clfs = {
    'GNB': GaussianNB(),
    'kNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(random_state=1234)
}

datasets = ['australian', 'balance', 'breastcan', 'cryotherapy', 'diabetes',
            'digit', 'ecoli4', 'german', 'glass2', 'heart', 'ionosphere',
            'liver', 'monkthree', 'shuttle-c0-vs-c4', 'sonar', 'soybean',
            'vowel0', 'waveform', 'wisconsin', 'yeast3']

n_datasets = len(datasets)
n_clfs = len(clfs)
n_splits = 5
n_repeats = 2

load = np.load('results.npy')
scores = load[0].T

alpha = 0.05

t_stat = np.zeros(shape=(n_clfs, n_clfs), dtype=float)
p_val = np.zeros(shape=(n_clfs, n_clfs), dtype=float)
adv = np.zeros(shape=(n_clfs, n_clfs), dtype=float)
sig = np.zeros(shape=(n_clfs, n_clfs), dtype=float)
s_better = np.zeros(shape=(n_clfs, n_clfs), dtype=float)

for i in range(n_clfs):
    for j in range(n_clfs):
        t_stat[i,j], p_val[i,j] = ttest_ind(scores[i], scores[j])

adv[t_stat > 0] = 1
sig[p_val <= alpha] = 1
s_better = adv*sig

print(f"t-stat:\n{t_stat}\np-value:\n{p_val}\nadvantage:\n{adv}\nsignificance:\n{sig}\nstat-better:\n{s_better}")
