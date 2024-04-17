#!/usr/bin/env python3

import numpy as np

from genetic_alg import generate_random_population

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC


np.random.seed(42)

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
population = generate_random_population(10, X_train.shape[1])


print(population)
print(X_train[:, population[0] == 1])
