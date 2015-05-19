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


def analyse_both(path, results_with_auto_adapt, results_without_auto_adapt):
    """
    Plots the best fitness of individuals in population throughout generations
    for both algorithms
    """

    run_counter = 1

    best_fitness_with = []
    average_fitness_with = []

    best_fitness_without = []
    average_fitness_without = []

    # get rid of the useless information for this analysis
    temp = []
    for accumulated_generations, pheno, problem in results_without_auto_adapt:
        temp.append(accumulated_generations)
    results_without_auto_adapt = deepcopy(temp)

    temp = []
    for sim_data, pheno, problem in results_with_auto_adapt:
        temp.append(sim_data[0])
    results_with_auto_adapt = deepcopy(temp)

    # TODO: fazer zip ao results_without_auto_adapt & results_with_auto_adapt
    # e fazer um plot para cada run

    for accumulated_generations in results_without_auto_adapt:
        for population in accumulated_generations:
            best_indiv = population[0]
            best_fitness_without.append(best_indiv[1])
            average_fitness_without.append(average_pop(population))

    for accumulated_generations in results_with_auto_adapt:
        for population in accumulated_generations:
            best_indiv = population[0]
            best_fitness_with.append(best_indiv[1])
            average_fitness_with.append(average_pop(population))

    plt.figure()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Comparison of fitness of best individual throughout generations')

    plt.plot(best_fitness_without, 'g', label="Best Without AD")
    plt.plot(average_fitness_without, 'g.', label="Averange Without AD")
    plt.plot(best_fitness_with, 'r', label="Best With AD")
    plt.plot(average_fitness_with, 'r.', label="Averange With AD")

    plt.legend(framealpha=0.5)
    plt.savefig(path + "/run_" + str(run_counter) + "/comparison.png", bbox_inches='tight')

    return

def plot_generations(path, accumulated_generations, title):
    """
    Plots the best and average fitness of individuals in population throughout generations
    """

    plt.figure()
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(title)

    best_fitness = []
    average_fitness = []

    for population in accumulated_generations:

        best_indiv = population[0]
        best_fitness.append(best_indiv[1])
        average_fitness.append(average_pop(population))

    plt.plot(best_fitness, 'g', label="Best")  # best individual
    plt.plot(average_fitness, 'r.', label="Average")  # average of individuals
    plt.legend(framealpha=0.5)
    plt.savefig(path, bbox_inches='tight')

    return


def analyse_regular(path, data):
    """
    função de análise estatística e apresentação de gráficos
    - analisar os melhores resultados e os resultados da média
    - analisar o efeito das alterações nos parâmetros
    """
    run_counter = 1

    for accumulated_generations, pheno, problem in data:

        # recor the results of this run in the appropriate directory
        best = best_pop(accumulated_generations[-1])
        write_file(path + "/run_" + str(run_counter) + "/WOAD.txt", display(best, pheno, problem))
        plot_generations(path + "/run_" + str(run_counter) + "/WOAD.png", accumulated_generations, 'Without AD')
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

        best = best_pop(accumulated_generations[-1])
        write_file(path + "/run_" + str(run_counter) + "/WAD.txt", display(best, pheno, problem))
        plot_generations(path + "/run_" + str(run_counter) + "/WAD.png", accumulated_generations, 'With AD')

        # Differences throughout generations
        plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Number of Differences')
        plt.title('Number of differences throughout generations')
        plt.plot(accumulated_differences, 'b.')

        plt.savefig(path + "/run_" + str(run_counter) + "/differences.png", bbox_inches='tight')
        run_counter += 1

    return

def analyse_results(path, number_of_runs, results):

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
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    # create folders and subfolders
    for run_counter in range(1, number_of_runs + 1):
        try:
            os.makedirs(path + "/run_" + str(run_counter))
        except Exception as e:
            print(e)

    return


# Auxiliary
def write_file(path, content):

    f = open(path, "w")
    f.write(content)
    f.close()
    return
