"""
Topic: Auto-Adaptation in Evolutionary Strategies
Authors: António Lima & Paulo Pereira
Remarks: Our proposal of an Auto-Adapting Evolutionary Algorithm
"""

import os
import shutil
import json

from kp_1 import *
import numpy as np
import matplotlib.pyplot as plt


AD_color = ['g', 'r', 'b', 'c', 'm', 'y']
AD_labels = 'AD0', 'AD1', 'AD2', 'AD3', 'AD4', 'AD5'


def comparison_pie_plot(path, data, txt):
    """
    Plots a pie chart comparison of the # of times each AD found
    the highest best/average final solution for all simulations
    """
    # The slices will be ordered and plotted counter-clockwise.

    unzipped_data = list(zip(*data))
    num_sims = len(unzipped_data)
    sizes = [0] * len(data)
    for simulation_data in unzipped_data:
        m = max(simulation_data)
        max_indexes = [i for i, j in enumerate(simulation_data) if j == m]
        for i in max_indexes:
            sizes[i] += 1

    # percentages
    for i in range(len(sizes)):
        sizes[i] = round((sizes[i] / num_sims) * 100)

    # "explode" all slices
    explode = [0.1] * len(data)

    plt.figure()
    plt.title("Comparison of the # of times each AD found the highest " + txt + " final solution for all simulations")

    plt.pie(sizes, explode=tuple(explode), labels=AD_labels, colors=AD_color, autopct='%1.1f%%', shadow=True)

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
    width = 0.175  # the width of the bars

    plt.xlabel('Simulation')
    plt.ylabel('Fitness')
    plt.title('Comparison of ' + txt + ' final individual\'s fitness for all simulations')

    for ith_ad in range(len(data)):
        ax.bar(ind + width * ith_ad, data[ith_ad], width, color=AD_color[ith_ad], label=txt + ' With AD' + str(ith_ad))

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(path + "/bar_" + txt + ".png", bbox_inches='tight')
    plt.close()

    return


def analyse_comparing(path, results, num_ads):
    """
    Plots the best fitness of individuals in population throughout generations
    for both algorithms
    """

    final_best_fitness_ad = [list(copy.deepcopy([])) for _ in range(num_ads + 1)]
    final_average_fitness_ad = [list(copy.deepcopy([])) for _ in range(num_ads + 1)]

    # get rid of the useless information for this analysis
    accumulated_generations_array, accumulated_diffs_array, crossover_probs_array, mutation_probs_array, phenotype_array, problem_array = results

    best_fitness_ad = []
    average_fitness_ad = []

    for acc_gen_ad in accumulated_generations_array:

        temp_temp_best_fit = []
        temp_temp_average_fit = []

        for run_ad in acc_gen_ad:

            temp_best_fit = []
            temp_average_fit = []

            for generation_ad in run_ad:
                best_indiv = generation_ad[0]
                temp_best_fit.append(best_indiv[1])
                temp_average_fit.append(average_pop(generation_ad))

            temp_temp_best_fit.append(temp_best_fit)
            temp_temp_average_fit.append(temp_average_fit)

        best_fitness_ad.append(temp_temp_best_fit)
        average_fitness_ad.append(temp_temp_average_fit)

    temp1 = []
    temp2 = []
    for run_counter in range(1, len(best_fitness_ad[0]) + 1):

        run_i = str(run_counter)

        # ----
        # Plot Comparison of fitnesses
        plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Comparison of individual\'s fitness throughout generations')

        temp3 = []
        temp4 = []
        for ith_ad in range(num_ads + 1):
            plt.plot(best_fitness_ad[ith_ad][run_counter - 1], AD_color[ith_ad],
                     label="Best With AD" + str(ith_ad))
            plt.plot(average_fitness_ad[ith_ad][run_counter - 1], AD_color[ith_ad] + '.',
                     label="Averange With AD" + str(ith_ad))
            temp3.append(best_fitness_ad[ith_ad][run_counter - 1])
            temp4.append(average_fitness_ad[ith_ad][run_counter - 1])

        temp1.append(temp3)
        temp2.append(temp4)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(path + "run_" + run_i + "/comparison_fitness.png", bbox_inches='tight')
        plt.close()

        # ----
        # Plot Comparison of Probabilities
        plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Probability')
        plt.title('Comparison of Crossover and Mutation probabilities throughout generations')

        for ith_ad in range(num_ads + 1):
            plt.plot(crossover_probs_array[ith_ad][run_counter - 1], AD_color[ith_ad],
                     label="Crossover AD" + str(ith_ad))
            plt.plot(mutation_probs_array[ith_ad][run_counter - 1], AD_color[ith_ad] + '-.',
                     label="Mutation AD" + str(ith_ad))

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(path + "/run_" + run_i + "/comparison_probabilities.png", bbox_inches='tight')
        plt.close()

        # ----
        # Plot Comparison of Differences
        plt.figure()
        plt.xlabel('Generation')
        plt.ylabel('Probability')
        plt.title('Comparison of Differences throughout generations')

        for ith_ad in range(num_ads + 1):
            plt.plot(accumulated_diffs_array[ith_ad][run_counter - 1], AD_color[ith_ad] + '.',
                     label="AD" + str(ith_ad))

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(path + "/run_" + run_i + "/comparison_differences.png", bbox_inches='tight')
        plt.close()

    # ----
    # store the last best/average solution's fitness for each AD of each simulation
    for ith_ad in range(num_ads + 1):
        for run_counter in range(len(best_fitness_ad[0])):
            final_best_fitness_ad[ith_ad].append(temp1[run_counter][ith_ad][-1])
            final_average_fitness_ad[ith_ad].append(temp2[run_counter][ith_ad][-1])

    # ----
    # Compare the last best solution of each simulation is for each method
    comparison_bar_plot(path, final_best_fitness_ad, "Best")
    comparison_pie_plot(path, final_best_fitness_ad, "Best")

    # ----
    # Compare the last average solution of each simulation is for each method
    comparison_bar_plot(path, final_average_fitness_ad, "Average")
    comparison_pie_plot(path, final_average_fitness_ad, "Average")

    return


def plot_generations(path, accumulated_generations, title, ith_ad):
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

    plt.plot(best_fitness, AD_color[ith_ad], label="Best")  # best individual
    plt.plot(average_fitness, AD_color[ith_ad] + '.', label="Average")  # average of individuals

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

    return


def analyse_AD(path, data):
    """
    função de análise estatística e apresentação de gráficos
    - analisar os melhores resultados e os resultados da média
    - analisar o efeito das alterações nos parâmetros
    """

    accumulated_generations_array, accumulated_differences_array, crossover_probs_array, mutation_probs_array, pheno_array, problem_array = data

    for ith_ad in range(len(accumulated_generations_array)):

        ad_i = str(ith_ad)
        print("\n\n----- Analysing results with AD" + ad_i + " -----")

        for run_counter in range(len(accumulated_generations_array[ith_ad])):
            run_i = str(run_counter + 1)
            print("---------- run " + run_i)

            accumulated_generations = accumulated_generations_array[ith_ad][run_counter]
            accumulated_differences = accumulated_differences_array[ith_ad][run_counter]
            crossover_probs = crossover_probs_array[ith_ad][run_counter]
            mutation_probs = mutation_probs_array[ith_ad][run_counter]
            pheno = pheno_array[ith_ad][run_counter]
            problem = problem_array[ith_ad][run_counter]

            best = best_pop(accumulated_generations[-1])
            write_str_to_file(path + "/run_" + run_i + "/generations_AD" + ad_i + ".txt", display(best, pheno, problem))
            plot_generations(path + "/run_" + run_i + "/generations_AD" + ad_i + ".png", accumulated_generations,
                             'With AD' + ad_i, ith_ad)

            # Differences throughout generations
            plt.figure()
            plt.xlabel('Generation')
            plt.ylabel('Number of Differences')
            plt.title('Number of differences throughout generations')
            plt.plot(accumulated_differences, AD_color[ith_ad] + '.')
            plt.savefig(path + "/run_" + run_i + "/differences_AD" + ad_i + ".png", bbox_inches='tight')
            plt.close()

            # How the Crossover and Mutation probabilities varied
            plt.figure()
            plt.xlabel('Generation')
            plt.ylabel('Probability')
            plt.title('Progression of Crossover and Mutation probabilities throughout generations')
            plt.plot(crossover_probs, AD_color[ith_ad], label="Crossover")
            plt.plot(mutation_probs, AD_color[ith_ad] + '-.', label="Mutation")

            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.savefig(path + "/run_" + run_i + "/probabilities_AD" + ad_i + ".png", bbox_inches='tight')
            plt.close()

    return


def analyse_results(path, number_of_runs, number_of_ads, results):
    """
    This method is the pivot point for the analysis of our simulation's results
    including, but not liimited to, the best and average solution throught the
    generations of each run.
    """
    # Delete and re-create necessary directories
    renew_directories(path, number_of_runs)

    # Compare results of both methods for all runs
    print("\n\n----- Comparing results -----")
    analyse_comparing(path, results, number_of_ads)

    # Analyse the results of the Auto-Adapt methods
    analyse_AD(path, results)


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