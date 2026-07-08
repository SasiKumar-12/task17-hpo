import numpy as np
from sklearn.datasets import load_breast_cancer  # swap this for your real dataset later
from sklearn.model_selection import train_test_split

SEED = 42

def get_data():
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return X_trainval, X_test, y_trainval, y_test