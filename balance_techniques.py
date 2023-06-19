
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss, CondensedNearestNeighbour, TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where


# # UNDERSAMPLING

# UNDERSAMPLE SELECTING EXAMPLES TO KEEP

def undersample_nearmiss(X_train, y_train, neighbors=3):
    undersample = NearMiss(version=1, n_neighbors=neighbors)
    X_train_us, y_train_us = undersample.fit_resample(X_train, y_train)
    return X_train_us, y_train_us


def undersample_condensed_nn (X_train, y_train, neighbors=1):
    undersample = CondensedNearestNeighbour(n_neighbors=neighbors)
    X_train_us, y_train_us = undersample.fit_resample(X_train, y_train)
    return X_train_us, y_train_us



# UNDERSAMPLE SELECTING EXAMPLES TO DELETE

def undersample_tomek_links(X_train, y_train):
    undersample = TomekLinks()
    X_train_us, y_train_us = undersample.fit_resample(X_train, y_train)
    return X_train_us, y_train_us


def undersample_edited_nn(X_train, y_train, neighbors=3):
    undersample = EditedNearestNeighbours(n_neighbors=neighbors)
    X_train_us, y_train_us = undersample.fit_resample(X_train, y_train)
    return X_train_us, y_train_us



# # OVERSAMPLING

# SMOTE

def oversample_smote(X_train, y_train, sampling_strategy=1):
    oversample = SMOTE(sampling_strategy)
    X_train_os, y_train_os = oversample.fit_resample(X_train, y_train)
    return X_train_os, y_train_os
    

