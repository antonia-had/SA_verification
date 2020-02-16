import numpy as np
from SALib.analyze import delta
from SALib.sample import latin
from fishery import fish_game

problem = {
  'num_vars': 9,
  'names': ['a', 'b', 'c', 'd','h',
            'K','m','sigmaX','sigmaY'],
  'bounds': [[ 0.002, 2],
             [0.005, 1],
             [0.2, 1],
             [0.05, 0.2],
             [0.001, 1],
             [100, 5000],
             [0.1, 1.5],
             [0.001, 0.01],
             [0.001, 0.01]]
}

# Generate samples
X = latin.sample(problem, 1000)

# Run model (example)
Y = [list(fish_game(*X[i,:])) for i in range(len(X))]

Y_NPV=np.asarray(list(zip(*Y))[0])

# Perform analysis
results = delta.analyze(problem, X, Y_NPV, print_to_console=True)
