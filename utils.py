
import numpy as np
import statsmodels.stats.power as power
import scipy.stats as stats

def run_simulation(pars):

    ind_var_names,ind_vars,dep_var_names,n_experiments,min_samples = pars
    holdout,e,a,b,l,i,seed = ind_vars
    
    np.random.seed(seed)
    
    budget_df = []
    
    tp = 0                
    fp = 0
    tn = 0
    fn = 0

    ave_tp_samples = 0
    
    for n in range(n_experiments):

        passed,samples = run_experiment(0,a,b,l,holdout,min_samples)
        if passed:
            fp += 1
        else:
            tn += 1

        passed,samples = run_experiment(e,a,b,l,holdout,min_samples)
        if passed:
            tp += 1
            ave_tp_samples += samples            
        else:
            fn += 1
        
        budget_df += [ind_vars + [samples]]

    if tp > 0:
        ave_tp_samples /= float(tp)
    else:
        ave_tp_samples = np.nan
    
    dep_vars = [tp,fp,tn,fn,ave_tp_samples]

    print([x for x in zip(ind_var_names, ind_vars)] +
          [x for x in zip(dep_var_names, dep_vars)] + [('Samples Taken', np.mean(samples))])
    print()
    
    return dep_vars, budget_df

def run_experiment(effect_size, alpha, beta, budget, with_holdout, min_samples):

    full_exploratory = get_data(budget, effect_size)
    full_confirmatory = get_data(budget, effect_size)
    
    check = False
    for n in np.linspace(min_samples, budget, 5):
        
        exploratory = full_exploratory[:int(n)]
        confirmatory = full_confirmatory[:int(n)]
        
        significant = test(exploratory, alpha)
        
        if significant:
            
            if with_holdout:
                passes = test(confirmatory, alpha)
                return passes, n
            
            return True, n

    return False, n

def get_data(budget, effect_size):
    
    return np.random.normal(size=budget) + effect_size

def test(data, alpha):
    
    return stats.ttest_1samp(data, 0.0).pvalue < alpha
