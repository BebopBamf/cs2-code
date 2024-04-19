import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

np.random.seed(42)

def generate_random_population(num_individuals, num_features):
    return np.random.randint(0, 2, (num_individuals, num_features))


def distance_to_best(individual, X_data, y_data, clf=SVC()):
    if np.sum(individual) == 0:
        return 0

    X_data = X_data[:, individual == 1]

    clf.fit(X_data, y_data)
    return roc_auc_score(y_data, clf.predict(X_data))


def calculate_fitness(population, X_data, y_data):
    return np.array(
        [distance_to_best(individual, X_data, y_data) for individual in population]
    )


# def genetic_feature_selection(data, val, n_features=10):
