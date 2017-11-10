import random

# probabilities must be a dictionary. If the entries are not summing
# up to one, they are normalized.

def simulateDiscreteDistribution(probabilities):
    total = 0
    for i,p in probabilities.items() :
        total += p
    if (total != 1.0) :
        print('Warning: probabilities do not sum up to one. They will be normalized')
    for i,p in probabilities.items() :
        probabilities[i] = p / total
    r = random.random() 
    total = 0 
    for i,p in probabilities.items() :
        total += p
        if (r < total) :
            return i
