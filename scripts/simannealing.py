# Example setup for local search based feature selection

import numpy as np
import random
import math

# a classifier that parses ROC score would be implemented here
def classifier(X, y, selected_features):
    roc = np.random()

    return roc

# In any local search based solution, we need an objective function, which informs the direction
# of the classifier.
# In this case, all this function needs to do is call our actual classifier and confirm that the
# subset is valid.

def objective(X, y, selected_features):
    if sum(selected_features) == 0: # an empty set of features is invalid
        return 0
    
    return classifier(X, y, selected_features)

# Essentially, this just finds adjacent features to the previously selected features.
def neighbour(current_features):
    neighbour = current_features.copy()
    index = random.randint(0, len(current_features) -1)
    neighbour[index] = 1 - neighbour[index]

    return neighbour

# Our acceptance logic (whether or not we accept a new subset), is fairly simple, if the new cost
# is less, we accept, and if its not, we accept based off of e ^ newcost - oldcost / temp

def acceptance_probability(oldcost, newcost, temp):
    if newcost > oldcost:
        return 1
    else:
        math.exp((newcost-oldcost)/temp)

# Here is our actual simmulated annealing algorithm, at the beginning, this algorithm will accept
# more agregious and volitile shifts in cost, however will decline options more frequently as we
# cool, and the algorithm ideally appraoches global minimum.

def simmulated_annealing(X, y):
    num_features = X.shape[1]
    current_features = [random.choice([0,1]) for _ in range(num_features)]
    current_cost = objective(X, y, current_features)
    temp = 1
    cooling = .99
    min_temp = .01

    while temp > min_temp:
        next_features = neighbour(current_features)
        next_cost = objective(X, y, next_features)

        # We chose to accept a new subset based on whether or not our acceptance probability is
        # higher or lower than a random int to prevent the algorithm from becoming deterministic.
        if acceptance_probability(current_cost, next_cost, temp) > random.random():
            current_features = next_features
            current_cost = next_cost
        
        temperature *= cooling

    return current_features