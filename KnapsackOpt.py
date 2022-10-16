import mlrose_hiive
from mlrose_hiive import Knapsack, KnapsackGenerator
from mlrose_hiive import algorithms
import numpy as np
import pdb
import time
import pandas as pd


from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

def runKnapsackOpt(length=50):
    
    # Set parameters for each type of algo, for this problem
    # ML Rose docs used to obtain some defaults (# Look for defaults from https://mlrose.readthedocs.io/en/stable/source/decay.html)
    # Defaults further developed by hand tuning parameter ranges

    rhc_params = {}
    sa_params = {}
    ga_params = {}
    mimic_params = {}


    rhc_params["max_iter"] = 1000
    rhc_params["max_attempts"] = 3000
    rhc_params["max_restarts"] = 10
    rhc_params["max_restarts_list"] = [0,5,10,15,20,25]


    sa_params["max_iter"] = 5000
    sa_params["max_attempts"] = 1000
    sa_params["decay_rates"] = [0.001, 0.005, 0.1, 0.15, 0.2] # List to allow for parameter studies
    sa_params["decay_rate"] = 0.9 # Kept constant
    sa_params["min_temp"] = .001
    sa_params["init_temp"] = 1


    ga_params["max_iter"] = 5000
    ga_params["max_attempts"] = 100
    ga_params["pop_size"] = 200
    ga_params["mutation_prob"] = 0.1
    ga_params["pop_sizes"] = [50, 100,200,300, 400,500]


    mimic_params["max_iter"] = 5000
    mimic_params["max_attempts"] = 1000
    mimic_params["pop_size"] = 200
    mimic_params["keep_pct"] = 0.5
    mimic_params["pop_sizes"] = [50, 100,200,300, 400,500]

    # As per https://mlrose.readthedocs.io/en/stable/source/fitness.html
    # This will be the Knapsack Optimization Problem
    problem = KnapsackGenerator().generate(seed=1,max_item_count=length)
    # problem = mlrose_hiive.DiscreteOpt(length=length, fitness_fn=fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)
    
    # Conduct a parameter variation study by changing the starting temperature of SA

    # First, plot performance at each iteration as the algos develop
    # Find the best scores, times, and iterations as well.  Dump to csv

    rhc_best_obj = None
    sa_best_obj = None
    ga_best_obj = None
    mimic_best_obj = None

    rhc_time = None
    sa_time = None
    ga_time = None
    mimic_time = None

    print("Running Performance Experiment for Knapsack \n")

    # Random Hill Climbing, single run with hand tuned params
    # tmp, since we do not care about the optimal state when gathering metrics
    start = time.time()
    tmp,rhc_best_obj, rhc_curve = mlrose_hiive.random_hill_climb(problem=problem, 
                                                        restarts=rhc_params["max_restarts"],
                                                        max_iters=rhc_params["max_iter"], 
                                                        max_attempts=rhc_params["max_attempts"], 
                                                        random_state=1,
                                                        # state_fitness_callback=time_track_callback,
                                                        curve=True)
    end = time.time()
    rhc_time = end-start



    # Simulated Annealing, single run with hand tuned params
    decay_schdule = algorithms.ExpDecay(init_temp=sa_params["init_temp"], min_temp=sa_params["min_temp"], exp_const=sa_params["decay_rate"])
    # tmp, since we do not care about the optimal state when gathering metrics
    start = time.time()
    tmp,sa_best_obj, sa_curve = mlrose_hiive.simulated_annealing(problem=problem, 
                                                        schedule=decay_schdule, 
                                                        max_iters=sa_params["max_iter"], 
                                                        max_attempts=sa_params["max_attempts"], 
                                                        random_state=1,
                                                        curve=True)
    end = time.time()
    sa_time = end-start

    
    # Genetic Algo, single run with hand tuned params
    # tmp, since we do not care about the optimal state when gathering metrics
    start = time.time()
    tmp,ga_best_obj, ga_curve = mlrose_hiive.genetic_alg(problem=problem, 
                                                        pop_size= ga_params["pop_size"],
                                                        mutation_prob= ga_params["mutation_prob"],
                                                        max_iters=ga_params["max_iter"], 
                                                        max_attempts=ga_params["max_attempts"], 
                                                        random_state=1,
                                                        curve=True)
    end = time.time()
    ga_time = end-start

    # MIMIC, single run with hand tuned params
    # tmp, since we do not care about the optimal state when gathering metrics
    start = time.time()
    tmp,mimic_best_obj, mimic_curve = mlrose_hiive.mimic(problem=problem, 
                                                        pop_size= mimic_params["pop_size"],
                                                        keep_pct = mimic_params["keep_pct"],
                                                        max_iters=mimic_params["max_iter"], 
                                                        max_attempts=mimic_params["max_attempts"], 
                                                        random_state=1,
                                                        curve=True)
    end = time.time()
    mimic_time = end-start
    

    rhc_iter = np.arange(1, rhc_curve.shape[0]+1)
    sa_iter = np.arange(1, sa_curve.shape[0]+1)
    ga_iter = np.arange(1, ga_curve.shape[0]+1)
    mimic_iter = np.arange(1, mimic_curve.shape[0]+1)


    # Save aggregated data (best scores, total runtime, max iterations reached) per algo
    best_obj_scores_agg = [rhc_best_obj, sa_best_obj, ga_best_obj, mimic_best_obj]
    times_agg = [rhc_time, sa_time, ga_time, mimic_time]
    sizes_agg = [rhc_iter.shape[0], sa_iter.shape[0], ga_iter.shape[0], mimic_iter.shape[0]]


    agg_data_df = pd.DataFrame()
    agg_data_df["best scores"] = best_obj_scores_agg
    agg_data_df["times"] = times_agg
    agg_data_df["max iterations"] = sizes_agg
    agg_data_df.to_csv('./results/{}_{}.csv'.format('Knapsack', length))
    print("Saved aggregate for Knapsack \n")

    rhc_best_objs = []
    sa_best_objs = []
    ga_best_objs = []
    mimic_best_objs = []


    plt.style.use('bmh')
    plt.figure()
    plt.plot(rhc_iter, rhc_curve[:,0], label = 'RHC', color="magenta")
    plt.plot(sa_iter, sa_curve[:,0], label = 'SA', color="cyan")
    plt.plot(ga_iter, ga_curve[:,0], label = 'GA', color="green")
    plt.plot(mimic_iter, mimic_curve[:,0], label = 'MIMIC', color="purple")
    plt.title('Score vs Iterations for RO algorithims.  Problem: {}'.format("Knapsack"))
    plt.xlabel("Iterations")
    plt.ylabel("Best Fitness Value")
    plt.legend()
    plt.savefig('./graphs/{}_{}.png'.format("Knapsack_Performance", length))


    rhc_restart_scores = []

    # pdb.set_trace()

    print("Running RHC Exp for Knapsack \n")

    # First, conduct a study over varying restarts for RHC
    for restart in rhc_params["max_restarts_list"]:
        tmp,rhc_restart_score, tmp2 = mlrose_hiive.random_hill_climb(problem=problem, 
                                                            restarts=restart,
                                                            max_iters=rhc_params["max_iter"], 
                                                            max_attempts=rhc_params["max_attempts"], 
                                                            random_state=1,
                                                            curve=False)
        rhc_restart_scores.append(rhc_restart_score)

    plt.figure()
    plt.plot(rhc_params["max_restarts_list"], rhc_restart_scores, label = 'RHC', color="black")
    plt.title('Restarts vs Best Score.  Problem: {}'.format("Knapsack"))
    plt.xlabel("Num Restarts")
    plt.ylabel("Best Fitness Value")
    plt.legend()
    plt.savefig('./graphs/{}_{}.png'.format("Knapsack_RHC_Restart", length))   


    print("Running SA Exp for Knapsack \n")

    # Next, iterate over decay rates for SA
    sa_decay_scores = []
    for decay_rate in sa_params["decay_rates"]:
        decay_schdule = algorithms.ExpDecay(init_temp=sa_params["init_temp"], min_temp=sa_params["min_temp"], exp_const=decay_rate)
        # tmp, since we do not care about the optimal state when gathering metrics
        tmp,sa_decay_best_obj, tmp2 = mlrose_hiive.simulated_annealing(problem=problem, 
                                                            schedule=decay_schdule, 
                                                            max_iters=sa_params["max_iter"], 
                                                            max_attempts=sa_params["max_attempts"], 
                                                            random_state=1,
                                                            curve=False)
        sa_decay_scores.append(sa_decay_best_obj)


    plt.style.use('bmh')
    plt.figure()
    plt.plot(sa_params["decay_rates"], sa_decay_scores, label = 'SA', color="orange")
    plt.title('Decay Rate vs Final Score.  Problem: {}'.format("Knapsack"))
    plt.xlabel("Decay Rate")
    plt.ylabel("Best Fitness Value")
    plt.legend()
    plt.savefig('./graphs/{}_{}.png'.format("Knapsack_SA_Decay", length))        


    # Conduct a study of parameter variation for GA and MIMIC

    print("Running GA/MIMIC Exp for Knapsack \n")

    ga_pop_scores = []
    mimic_pop_scores = []

    for pop_size in ga_params["pop_sizes"]:
        # Genetic Algo, single run with hand tuned params
        # tmp, since we do not care about the optimal state when gathering metrics
        tmp,ga_pop_score, tmp2 = mlrose_hiive.genetic_alg(problem=problem, 
                                                            pop_size= pop_size,
                                                            mutation_prob= ga_params["mutation_prob"],
                                                            max_iters=ga_params["max_iter"], 
                                                            max_attempts=ga_params["max_attempts"], 
                                                            random_state=1,
                                                            curve=False)

        ga_pop_scores.append(ga_pop_score)

        # MIMIC, single run with hand tuned params
        # tmp, since we do not care about the optimal state when gathering metrics
        tmp, mimic_pop_score, tmp2 = mlrose_hiive.mimic(problem=problem, 
                                                            pop_size= pop_size,
                                                            keep_pct = mimic_params["keep_pct"],
                                                            max_iters=mimic_params["max_iter"], 
                                                            max_attempts=mimic_params["max_attempts"], 
                                                            random_state=1,
                                                            curve=False)

        mimic_pop_scores.append(mimic_pop_score)

    plt.figure()
    plt.plot(ga_params["pop_sizes"], ga_pop_scores, label = 'GA', color="red")
    plt.plot(ga_params["pop_sizes"], mimic_pop_scores, label = 'MIMIC', color="blue")
    plt.title('Pop Size vs Final Score.  Problem: {}'.format("Knapsack"))
    plt.xlabel("Population Size")
    plt.ylabel("Best Fitness Value")
    plt.legend()
    plt.savefig('./graphs/{}_{}.png'.format("Knapsack_PopSize", length))  

    print("All Knapsack Experiments Complete. Results saved.")