import numpy as np
from SALib.analyze import delta
from SALib.sample import latin
from fishery import fish_game
import matplotlib.pyplot as plt

# Set up dictionary with system parameters
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
nsamples = 1000
X = latin.sample(problem, nsamples)

# Run model for all samples
output = [list(fish_game(*X[j,:])) for j in range(nsamples)]

# Pick out objective of interest
Y=np.asarray(list(zip(*output))[0])

# Perform analysis
results = delta.analyze(problem, X, Y, print_to_console=True)

# Sort factors by importance
factors_sorted = np.argsort(results['delta'])[::-1]

for f in range(1, len(factors_sorted)):
    ntopfactors = f
    problem_reduced = {'num_vars': ntopfactors,
                       'names': [problem['names'][factors_sorted[i]] for i in range(ntopfactors)],
                       'bounds': [problem['bounds'][factors_sorted[i]] for i in range(ntopfactors)]}
    
    # Generate samples
    X_reduced = latin.sample(problem_reduced, nsamples)
    
    # Run model for all samples
    output_reduced = [[]]*nsamples
    for j in range (nsamples):
        kwargs = {problem['names'][factors_sorted[i]]:X_reduced[j,i] for i in range(ntopfactors)}
        output_reduced[j] = list(fish_game(**kwargs))
    
    # Pick out objective of interest
    Y_reduced=np.asarray(list(zip(*output_reduced))[0])
    plt.scatter(np.sort(Y),np.sort(Y_reduced))
    plt.plot(np.sort(Y),np.sort(Y))
    if np.corrcoef(np.sort(Y),np.sort(Y_reduced))[0][1] >= 0.9:
      break
