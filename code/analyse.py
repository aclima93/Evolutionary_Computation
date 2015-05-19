"""
Topic: Auto-Adaptation in Evolutionary Strategies
Authors: António Lima & Paulo Pereira
Remarks: Our proposal of an Auto-Adapting Evolutionary Algorithm
"""

from kp_1 import *
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import shutil
import json


def analyse_both(path, results_wad, results_woad):
    """
    Plots the best fitness of individuals in population throughout generations
    for both algorithms
    """
    run_counter = 1

    best_fitness_wad = []
    average_fitness_wad = []

    best_fitness_woad = []
    average_fitness_woad = []

    # get rid of the useless information for this analysis
    temp = []
    for accumulated_generations, pheno, problem in results_woad:
        temp.append(accumulated_generations)
    results_woad = deepcopy(temp)

    temp = []
    for sim_data, pheno, problem in results_wad:
        temp.append(sim_data[0])
    results_wad = deepcopy(temp)

    clean_results = list(zip(results_woad, results_wad))

    for acc_gen_woad, acc_gen_wad in clean_results:

        for population_woad in acc_gen_woad:
            best_indiv = population_woad[0]
            best_fitness_woad.append(best_indiv[1])
            average_fitness_woad.append(average_pop(population_woad))

        for population_wad in acc_gen_wad:
            best_indiv = population_wad[0]
            best_fitness_wad.append(best_indiv[1])
            average_fitness_wad.append(average_pop(population_wad))

        plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Comparison of individual\'s fitness throughout generations')

        plt.plot(best_fitness_woad, 'g', label="Best Without AD")
        plt.plot(average_fitness_woad, 'g.', label="Averange Without AD")
        plt.plot(best_fitness_wad, 'r', label="Best With AD")
        plt.plot(average_fitness_wad, 'r.', label="Averange With AD")

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(path + "/run_" + str(run_counter) + "/comparison.png", bbox_inches='tight')
        plt.close()

        run_counter += 1

    return


def plot_generations(path, accumulated_generations, title):
    """
    Plots the best and average fitness of individuals in population throughout generations
    """
    best_fitness = []
    average_fitness = []

    for population in accumulated_generations:
        best_indiv = population[0]
        best_fitness.append(best_indiv[1])
        average_fitness.append(average_pop(population))

    plt.figure()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(title)
    plt.plot(best_fitness, 'g', label="Best")  # best individual
    plt.plot(average_fitness, 'r.', label="Average")  # average of individuals
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    return


def analyse_regular(path, data):
    """
    função de análise estatística e apresentação de gráficos
    - analisar os melhores resultados e os resultados da média
    - analisar o efeito das alterações nos parâmetros
    """
    run_counter = 1

    for accumulated_generations, pheno, problem in data:

        # record the results of this simulate in the appropriate directory
        best = best_pop(accumulated_generations[-1])
        write_str_to_file(path + "/run_" + str(run_counter) + "/WOAD.txt", display(best, pheno, problem))
        plot_generations(path + "/run_" + str(run_counter) + "/WOAD.png", accumulated_generations, 'Without AD')
        plt.close()

        run_counter += 1

    return


def analyse_auto_adapt(path, data):
    """
    função de análise estatística e apresentação de gráficos
    - analisar os melhores resultados e os resultados da média
    - analisar o efeito das alterações nos parâmetros
    """
    run_counter = 1

    for sim_data, pheno, problem in data:
        accumulated_generations = sim_data[0]
        accumulated_differences = sim_data[1]
        crossover_probs = sim_data[2]
        mutation_probs = sim_data[3]

        best = best_pop(accumulated_generations[-1])
        write_str_to_file(path + "/run_" + str(run_counter) + "/WAD.txt", display(best, pheno, problem))
        plot_generations(path + "/run_" + str(run_counter) + "/WAD.png", accumulated_generations, 'With AD')

        # Differences throughout generations
        plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Number of Differences')
        plt.title('Number of differences throughout generations')
        plt.plot(accumulated_differences, 'b.')
        plt.savefig(path + "/run_" + str(run_counter) + "/differences.png", bbox_inches='tight')
        plt.close()

        # How the Crossover and Mutation probabilities varied
        plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Probability')
        plt.title('Progression of Crossover and Mutation probabilities throughout generations')
        plt.plot(crossover_probs, 'b', label="Crossover")
        plt.plot(mutation_probs, 'r', label="Mutation")

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(path + "/run_" + str(run_counter) + "/probabilities.png", bbox_inches='tight')
        plt.close()

        run_counter += 1

    return


def analyse_results(path, number_of_runs, results):
    """
    This method is the pivot point for the analysis of our simulation's results
    including, but not liimited to, the best and average solution throught the
    generations of each run.
    """
    # Delete and re-create necessary directories
    renew_directories(path, number_of_runs)

    # Compare results of both methods for all runs
    print("\n\n----- Comparing results -----")
    analyse_both(path, results[0], results[1])

    # Analyse the results with Auto-Adapt
    print("\n\n----- Analysing results with Auto-Adapt -----")
    analyse_auto_adapt(path, results[0])

    # Analyse the results without Auto-Adapt
    print("\n\n----- Analysing results without Auto-Adapt -----")
    analyse_regular(path, results[1])


def renew_directories(path, number_of_runs):
    # delete folders, subfolders and content
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    # create folders and subfolders
    for run_counter in range(1, number_of_runs + 1):
        try:
            os.makedirs(path + "/run_" + str(run_counter))
        except Exception as e:
            print(e)

    return


# Auxiliary function
def write_str_to_file(path, content):
    f = open(path, "w")
    f.write(content)
    f.close()
    return


# Auxiliary function
def write_dic_to_file(path, dic):
    f = open(path, "w")
    json.dump(dic, f)
    f.close()
    return