"""
Topic: Auto-Adaptation in Evolutionary Strategies
Authors: António Lima & Paulo Pereira
Remarks: Our proposal of an Auto-Adapting Evolutionary Algorithm
"""

from kp_1 import *
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
import shutil
import json


def comparison_pie_plot(path, data, txt):
    """
    Plots a pie chart comparison of the # of times each AD found
    the highest best/average final solution for all simulations
    """
    # The slices will be ordered and plotted counter-clockwise.
    labels = 'WOAD', 'AD1', 'AD2'
    colors = ['green', 'red', 'blue']

    unzipped_data = list(zip(data[0], data[1], data[2]))
    num_sims = len(unzipped_data)
    sizes = [0, 0, 0]
    for simulation_data in unzipped_data:
        m = max(simulation_data)
        max_indexes = [i for i, j in enumerate(simulation_data) if j == m]
        for i in max_indexes:
            sizes[i] += 1

    # percentages
    for i in range(len(sizes)):
        sizes[i] = round((sizes[i] / num_sims) * 100)

    # only "explode" the best slices
    m = max(sizes)
    max_indexes = [i for i, j in enumerate(sizes) if j == m]
    explode = [0, 0, 0]
    for i in max_indexes:
        explode[i] = 0.1

    plt.figure()
    plt.title("Comparison of the # of times each AD found the highest " + txt + " final solution for all simulations")

    plt.pie(sizes, explode=tuple(explode), labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)

    plt.axis('equal')  # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.savefig(path + "/pie_" + txt + ".png", bbox_inches='tight')
    plt.close()

    return

def comparison_bar_plot(path, data, txt):
    """
    Plots a bar chart comparison of each AD's highest best/average final solution for all simulations
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ind = np.arange(len(data[0]))  # the x locations for the groups
    width = 0.2  # the width of the bars

    plt.xlabel('Simulation')
    plt.ylabel('Fitness')
    plt.title('Comparison of ' + txt + ' final individual\'s fitness for all simulations')

    ax.bar(ind, data[0], width, color='green', label=txt + ' Without AD')
    ax.bar(ind + width, data[1], width, color='red', label=txt + ' With AD1')
    ax.bar(ind + width * 2, data[2], width, color='blue', label=txt + ' With AD2')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(path + "/bar_" + txt + ".png", bbox_inches='tight')
    plt.close()

    return


def analyse_comparing(path, results):
    """
    Plots the best fitness of individuals in population throughout generations
    for both algorithms
    """
    run_counter = 1
    results_woad, results_wad1, results_wad2 = results

    final_best_fitness_wad2 = []
    final_average_fitness_wad2 = []

    final_best_fitness_wad1 = []
    final_average_fitness_wad1 = []

    final_best_fitness_woad = []
    final_average_fitness_woad = []

    # get rid of the useless information for this analysis
    temp = []
    for accumulated_generations, pheno, problem in results_woad:
        temp.append(accumulated_generations)
    results_woad = deepcopy(temp)

    temp = []
    for sim_data, pheno, problem in results_wad1:
        temp.append(sim_data[0])
    results_wad1 = deepcopy(temp)

    temp = []
    for sim_data, pheno, problem in results_wad2:
        temp.append(sim_data[0])
    results_wad2 = deepcopy(temp)

    clean_results = list(zip(results_woad, results_wad1, results_wad2))

    for acc_gen_woad, acc_gen_wad1, acc_gen_wad2 in clean_results:

        run_i = str(run_counter)

        best_fitness_wad1 = []
        average_fitness_wad1 = []

        best_fitness_wad2 = []
        average_fitness_wad2 = []

        best_fitness_woad = []
        average_fitness_woad = []

        for population_woad in acc_gen_woad:
            best_indiv = population_woad[0]
            best_fitness_woad.append(best_indiv[1])
            average_fitness_woad.append(average_pop(population_woad))

        for population_wad1 in acc_gen_wad1:
            best_indiv = population_wad1[0]
            best_fitness_wad1.append(best_indiv[1])
            average_fitness_wad1.append(average_pop(population_wad1))

        for population_wad2 in acc_gen_wad2:
            best_indiv = population_wad2[0]
            best_fitness_wad2.append(best_indiv[1])
            average_fitness_wad2.append(average_pop(population_wad2))

        plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Comparison of individual\'s fitness throughout generations')

        plt.plot(best_fitness_woad, 'g', label="Best Without AD")
        plt.plot(average_fitness_woad, 'g.', label="Averange Without AD")
        plt.plot(best_fitness_wad1, 'r', label="Best With AD1")
        plt.plot(average_fitness_wad1, 'r.', label="Averange With AD1")
        plt.plot(best_fitness_wad2, 'b', label="Best With AD2")
        plt.plot(average_fitness_wad2, 'b.', label="Averange With AD2")

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(path + "/run_" + run_i + "/comparison.png", bbox_inches='tight')
        plt.close()

        run_counter += 1

        # store the last best solution'sfitness
        final_best_fitness_woad.append(best_fitness_woad[-1])
        final_average_fitness_woad.append(average_fitness_woad[-1])

        final_best_fitness_wad1.append(best_fitness_wad1[-1])
        final_average_fitness_wad1.append(average_fitness_wad1[-1])

        final_best_fitness_wad2.append(best_fitness_wad2[-1])
        final_average_fitness_wad2.append(average_fitness_wad2[-1])

    # ----
    # Compare the last best solution of each simulation is for each method
    final_data = [final_best_fitness_woad, final_best_fitness_wad1, final_best_fitness_wad2]
    comparison_bar_plot(path, final_data, "Best")
    comparison_pie_plot(path, final_data, "Best")

    # ----
    # Compare the last average solution of each simulation is for each method
    final_data = [final_average_fitness_woad, final_average_fitness_wad1, final_average_fitness_wad2]
    comparison_bar_plot(path, final_data, "Average")
    comparison_pie_plot(path, final_data, "Average")

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


def analyse_standard(path, data):
    """
    função de análise estatística e apresentação de gráficos
    - analisar os melhores resultados e os resultados da média
    - analisar o efeito das alterações nos parâmetros
    """

    run_counter = 1

    for accumulated_generations, pheno, problem in data:

        run_i = str(run_counter)

        # record the results of this simulate in the appropriate directory
        best = best_pop(accumulated_generations[-1])

        write_str_to_file(path + "/run_" + run_i + "/WOAD.txt", display(best, pheno, problem))
        plot_generations(path + "/run_" + run_i + "/WOAD.png", accumulated_generations, 'Without AD')

        run_counter += 1

    return


def analyse_AD(path, data, ad_type):
    """
    função de análise estatística e apresentação de gráficos
    - analisar os melhores resultados e os resultados da média
    - analisar o efeito das alterações nos parâmetros
    """

    ad_i = str(ad_type)
    run_counter = 1

    for sim_data, pheno, problem in data:

        run_i = str(run_counter)

        accumulated_generations = sim_data[0]
        accumulated_differences = sim_data[1]
        crossover_probs = sim_data[2]
        mutation_probs = sim_data[3]

        best = best_pop(accumulated_generations[-1])
        write_str_to_file(path + "/run_" + run_i + "/AD" + ad_i + ".txt", display(best, pheno, problem))
        plot_generations(path + "/run_" + run_i + "/AD" + ad_i + ".png", accumulated_generations, 'With AD' + ad_i)

        # Differences throughout generations
        plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Number of Differences')
        plt.title('Number of differences throughout generations')
        plt.plot(accumulated_differences, 'b.')
        plt.savefig(path + "/run_" + run_i + "/differences" + ad_i + ".png", bbox_inches='tight')
        plt.close()

        # How the Crossover and Mutation probabilities varied
        plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Probability')
        plt.title('Progression of Crossover and Mutation probabilities throughout generations')
        plt.plot(crossover_probs, 'b', label="Crossover")
        plt.plot(mutation_probs, 'r', label="Mutation")

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(path + "/run_" + run_i + "/probabilities_AD" + ad_i + ".png", bbox_inches='tight')
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
    analyse_comparing(path, results)

    # Analyse the results with Auto-Adapt1
    print("\n\n----- Analysing results with AD1 -----")
    analyse_AD(path, results[1], 1)

    # Analyse the results with Auto-Adapt1
    print("\n\n----- Analysing results with AD2 -----")
    analyse_AD(path, results[2], 2)

    # Analyse the results without Auto-Adapt
    print("\n\n----- Analysing results without AD -----")
    analyse_standard(path, results[0])


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