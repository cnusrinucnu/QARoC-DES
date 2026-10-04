import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import sklearn.preprocessing as preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare
import seaborn as sns

from deslib.dcs import LCA
from deslib.dcs import MLA
from deslib.dcs import OLA
from deslib.dcs import MCB
from deslib.dcs import Rank
from deslib.des import KNORAE, KNORAU, KNOP, METADES, DESKNN, DESClustering
from deslib.static.oracle import Oracle
from deslib.static.single_best import SingleBest
from deslib.util.datasets import make_P2

from scipy.spatial.distance import cdist
from scipy.stats import mode
from scipy.stats import binom

import scipy.io as sio
import time
import os
import warnings
import math
from itertools import combinations

from mealpy import Problem, Optimizer, Multitask, FloatVar, PSO, WOA, GA, DE,GWO
from mealpy.utils.problem import Problem

from multiprocessing import Pool


warnings.filterwarnings("ignore")

import config
from des_mha import DES_MHA
import helpers

from asydes import ASyDES
