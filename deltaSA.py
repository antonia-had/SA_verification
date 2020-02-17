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

defaultvalues = np.array([0.005, 0.5, 0.5, 0.1, 0.1, 2000, 0.7, 0.004, 0.004])

# Generate samples
nsamples = 1000
X_Set1 = latin.sample(problem, nsamples) # This is Set 1

# Run model for all samples
output = [fish_game(*X_Set1[j,:]) for j in range(nsamples)]

# Perform analysis
results = delta.analyze(problem, X_Set1, np.asarray(output), print_to_console=True)

# Sort factors by importance
factors_sorted = np.argsort(results['delta'])[::-1]

# Set up DataFrame of default values to use for experiment
X_defaults = np.tile(defaultvalues,(nsamples, 1))

# Create initial Sets 2 and 3
X_Set2 = np.copy(X_defaults)
X_Set3 = np.copy(X_Set1)

for f in range(1, len(factors_sorted)+1):
    ntopfactors = f
    
    for i in range(ntopfactors): #Loop through all important factors
        X_Set2[:,factors_sorted[i]] = X_Set1[:,factors_sorted[i]] #Fix use samples for important
        X_Set3[:,factors_sorted[i]] = X_defaults[:,factors_sorted[i]] #Fix important to defaults
    
    # Run model for all samples    
    output_Set2 = [fish_game(*X_Set2[j,:]) for j in range(nsamples)]
    output_Set3 = [fish_game(*X_Set3[j,:]) for j in range(nsamples)]
    
    # Calculate coefficients of correlation
    coefficient_S1_S2 = np.corrcoef(output,output_Set2)[0][1]
    coefficient_S1_S3 = np.corrcoef(output,output_Set3)[0][1]
    
    # Plot outputs and correlation
    fig =  plt.figure(figsize=(18,12))
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(output,output)
    ax1.scatter(output,output_Set2)
    ax1.set_xlabel("Set 1",fontsize=14)
    ax1.set_ylabel("Set 2",fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.set_title('Set 1 vs Set 2 - ' + str(f) + ' top factors',fontsize=20)
    ax1.text(0.05,0.95,'R= '+"{0:.3f}".format(coefficient_S1_S2),transform = ax1.transAxes,fontsize=16)
    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(output,output)
    ax2.scatter(output,output_Set3)
    ax2.set_xlabel("Set 1",fontsize=14)
    ax2.set_ylabel("Set 3",fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_title('Set 1 vs Set 3 - ' + str(f) + ' top factors',fontsize=20)
    ax2.text(0.05,0.95,'R= '+"{0:.3f}".format(coefficient_S1_S3),transform = ax2.transAxes,fontsize=16)
    plt.savefig(str(f)+'_topfactors.png')
    plt.close()