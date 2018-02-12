
import numpy as np
import pandas as pd

import utils

from multiprocessing import Pool

n_reps = 10
n_experiments = 1000

min_samples = 100

effect_sizes = [0.1, 0.25, 0.5]
alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0] 
betas = [0.8] 
limits = [200, 500, 1000]

main_df = []
budget_df = []

ind_var_names = ['Holdout',
                 'True Effect Size',
                 'Theoretical Alpha',
                 'Theoretical Beta',
                 'Sample Budget',
                 'Repetition',
                 'Seed']

dep_var_names = ['True Positive',
                 'False Positive',
                 'True Negative',
                 'False Negative',
                 'Average True Positive Samples']

pars = []

for holdout in [0, 1]:
    for e in effect_sizes:
        for a in alphas:
            for b in betas:
                for l in limits:                    
                    for i in range(n_reps):

                        seed = np.random.randint(2**32 - 1)
                        
                        ind_vars = [holdout,e,a,b,l,i,seed]
                        
                        pars += [(ind_var_names,ind_vars,dep_var_names,n_experiments,min_samples)]

p = Pool(4)                        
#results = [x for x in map(utils.run_simulation, pars)]
results = [x for x in p.map(utils.run_simulation, pars)]

IND_VARS = 1
DEP_VARS = 0
BUDGET = 1

main_df = []
budget_df = []
for i,r in enumerate(results):

    main_df += [pars[i][IND_VARS] + results[i][DEP_VARS]]    
    budget_df += results[i][BUDGET]

main_df = pd.DataFrame(main_df)
main_df.columns = ind_var_names + dep_var_names

main_df['Simulated Alpha'] = main_df['False Positive'] / float(n_experiments)
main_df['Simulated Beta'] = main_df['True Positive'] / float(n_experiments)

budget_df = pd.DataFrame(budget_df)
budget_df.columns = ind_var_names + ['Samples Taken']

main_df.to_csv('ttest_main_df.csv', index = False)
budget_df.to_csv('ttest_budget_df.csv', index = False)
