"""
Topic: Auto-Adaptation in Evolutionary Strategies
Authors: António Lima & Paulo Pereira
Remarks: Our proposal of an Auto-Adapting Evolutionary Algorithm
Notation Clarifications:
- AD stands for (Auto-Adaptive)
- AD0: default implementation provided by professor Ernesto
- AD1:
- AD2:
- AD3:
- AD4:
"""

from time import *
from random import *
from analyse import *


def best_reff_pop_fit(refference_window):
    """
    returns the average best fitness based on all individuals in the refference population
    """
    return sum([best_pop(population)[1] for population in refference_window]) / len(refference_window)


def average_reff_pop_fit(refference_window):
    """
    returns the average fitness based on all individuals in the refference population
    """
    return sum([average_pop(population) for population in refference_window]) / len(refference_window)


def best_indiv(population):
    """
    returns the best individual in the population
    """
    return best_pop(population)


def average_indiv(population, fitness_func):
    """
    returns the average individual based on all individuals in the population
    """
    individuals = [indiv for indiv, fit in population]

    num_individuals = len(individuals)
    average_individual = [0] * len(individuals[0])  # individual with only zeros
    indexes = range(len(individuals[0]))
    # sum all the solutions
    for indiv in individuals:
        for i in indexes:
            average_individual[i] += indiv[i]

    # divide by number of individuals in population
    average_individual = [round(average_individual[i] / num_individuals) for i in range(len(average_individual))]

    return [average_individual, fitness_func(average_individual)]


def best_reff_pop(refference_window, fitness_func):
    """
    returns the average population based on the ith best individual of all populations
    """
    ith_pairings = list(zip(*refference_window))
    return list(ith_pairings[0])


def average_reff_pop(refference_window, fitness_func):
    """
    returns the average population based on the ith average individual of all populations
    """
    ith_pairings = list(zip(*refference_window))
    return [average_indiv(ith_pair, fitness_func) for ith_pair in ith_pairings]


def AD5_fitness(cur_population, refference_window, fitness_func):
    """
    Analyses the populations stored in the refference window and retrieves the best individual of all.
    Then we also fetch the best individual for the current population
    and return the number of differences between them.
    """

    # get best average population from refference_window
    best_reff_population = best_reff_pop(refference_window, fitness_func)

    # get average individual from best population
    best_reff_individual = best_indiv(best_reff_population)

    # get best individual from current population
    best_individual = best_indiv(cur_population)

    fit1 = best_individual[1]
    fit2 = best_reff_individual[1]
    return fit1, fit2

def AD4_fitness(cur_population, refference_window):
    """
    Analyses the populations stored in the refference window and creates a best fitness.
    Then we also derive the best fitness for the current population and return the difference between them.
    """
    fit1 = best_pop(cur_population)[1]
    fit2 = best_reff_pop_fit(refference_window)
    return fit1, fit2


def AD3_fitness(cur_population, refference_window, fitness_func):
    """
    Analyses the populations stored in the refference window and retrieves the best individual of all.
    Then we also fetch the best individual for the current population
    and return the number of differences between them.
    """

    # get best average population from refference_window
    best_reff_population = best_reff_pop(refference_window, fitness_func)

    # get average individual from best population
    best_reff_individual = average_indiv(best_reff_population, fitness_func)

    # get best individual from current population
    best_individual = best_indiv(cur_population)

    fit1 = best_individual[1]
    fit2 = best_reff_individual[1]
    return fit1, fit2


def AD2_fitness(cur_population, refference_window):
    """
    Analyses the populations stored in the refference window and creates an average fitness.
    Then we also derive the average fitness for the current population and return the difference between them.
    """
    fit1 = average_pop(cur_population)  # sim, isto devolve a fitness média de uma população
    fit2 = average_reff_pop_fit(refference_window)
    return fit1, fit2


def AD1_fitness(cur_population, refference_window, fitness_func):
    """
    Analyses the populations stored in the refference window and creates an average population.
    From this average population we derive an average individual.
    Then we also derive the average individual for the current population
    and return the number of differences between them.
    """

    # get average population from refference_window
    average_reff_population = average_reff_pop(refference_window, fitness_func)

    # get average individual from average population
    average_reff_individual = average_indiv(average_reff_population, fitness_func)

    # get average individual from current population
    average_individual = average_indiv(cur_population, fitness_func)

    fit1 = average_individual[1]
    fit2 = average_reff_individual[1]
    return fit1, fit2


def AD(ad_type, numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation,
       sel_survivors, fitness_func, refference_window_size, refference_window):
    """
    funcao AD
    - variação da sea fornecida pelo prof. Ernesto mas que aplica os conceitos que estamos a estudar:
    --- variação do tamanho da população, prob. de mutação e prob. de crossover
    returns the best individual in the final generation as well as all generations and the list of all differences
    """

    accumulated_generations = []
    accumulated_diffs = []
    crossover_probs = []
    mutation_probs = []
    difference_history = 0
    len_reff_window = len(refference_window)

    # speedup by refferencing faster
    refference_window_append = refference_window.append
    refference_window_remove = refference_window.remove
    accumulated_generations_append = accumulated_generations.append
    accumulated_diffs_append = accumulated_diffs.append
    crossover_probs_append = crossover_probs.append
    mutation_probs_append = mutation_probs.append
    

    # inicializa população: indiv = (cromo,fit)
    populacao = gera_pop(size_pop, size_cromo)
    accumulated_generations_append(populacao)

    # avalia população
    populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]

    for i in range(numb_generations):

        # selecciona progenitores
        mate_pool = sel_parents(populacao)

        # Variation
        # ------ Crossover
        progenitores = []
        for j in range(0, size_pop - 1, 2):
            cromo_1 = mate_pool[j]
            cromo_2 = mate_pool[j + 1]
            filhos = recombination(cromo_1, cromo_2, prob_cross)
            progenitores.extend(filhos)
            # ------ Mutation

        descendentes = []
        descendentes_append = descendentes.append

        for indiv, fit in progenitores:
            novo_indiv = mutation(indiv, prob_mut)
            descendentes_append((novo_indiv, fitness_func(novo_indiv)))

        # added previous generation to refference_window and remove the oldest (or just append until we have enough)
        if len_reff_window == refference_window_size:
            refference_window_remove(refference_window[0])
            len_reff_window -= 1
        refference_window_append(populacao)
        len_reff_window += 1

        # New population
        populacao = sel_survivors(populacao, descendentes)
        # Avalia nova _população
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]

        # store the population
        accumulated_generations_append(populacao)

        # only apply our strategies after the refference_window is full (ignoring the initial abrupt changes)
        if len_reff_window == refference_window_size:

            # perform auto-adaptive changes according to refference window
            if ad_type == 1:
                fit1, fit2 = AD1_fitness(populacao, refference_window, fitness_func)

            elif ad_type == 2:
                fit1, fit2 = AD2_fitness(populacao, refference_window)

            elif ad_type == 3:
                fit1, fit2 = AD3_fitness(populacao, refference_window, fitness_func)

            elif ad_type == 4:
                fit1, fit2 = AD4_fitness(populacao, refference_window)

            else:  # ad_type == 5:

                fit1, fit2 = AD5_fitness(populacao, refference_window, fitness_func)


            diffs = abs(fit1 - fit2)
            accumulated_diffs_append(diffs)

            # ----
            # if the difference ratio falls below the threshhold alter the crossover and mutation probabilities
            if diffs <= ACTIVATION_THRESHOLD:
                difference_history += 1
            else:
                difference_history = 0  # reset, must be consecutive

            if difference_history == CONSECUTIVE_ACTIVATIONS:

                # reset the counter, let it build up again
                difference_history = 0

                if (prob_cross - CROSSOVER_STEP) >= CROSSOVER_BOUND:
                    prob_cross -= CROSSOVER_STEP

                if (prob_mut + MUTATION_STEP) <= MUTATION_BOUND:
                    prob_mut += MUTATION_STEP
        else:
            accumulated_diffs_append(-1)  # para podermos mostrar o gráfico de todas as probabilidades

        crossover_probs_append(prob_cross)
        mutation_probs_append(prob_mut)

    return accumulated_generations, accumulated_diffs, crossover_probs, mutation_probs


def simulate(auto_adapt_type, problem, size_items):
    """
    Runs one simulation for the provided parameters.
    Returns the simulation's results
    """

    refference_window_size = WINDOW_SIZE
    refference_window = []

    num_generations = NUM_GENERATIONS
    population_size = POPULATION_SIZE
    prob_crossover = PROB_CROSSOVER
    prob_mutation = PROB_MUTATION

    if auto_adapt_type != 0:
        # AD1(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination,
        # mutation, sel_survivors, fitness_func, refference_window_size, refference_window)
        accumulated_generations, accumulated_diffs, crossover_probs, mutation_probs = AD(auto_adapt_type,
                                                                                         num_generations,
                                                                                         population_size, size_items,
                                                                                         prob_mutation, prob_crossover,
                                                                                         tour_sel(3), one_point_cross,
                                                                                         muta_bin,
                                                                                         sel_survivors_elite(0.02),
                                                                                         merito(problem),
                                                                                         refference_window_size,
                                                                                         refference_window)
    else:
        # sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination,
        # mutation, sel_survivors, fitness_func)
        accumulated_generations, accumulated_diffs, crossover_probs, mutation_probs = sea(num_generations,
                                                                                          population_size, size_items,
                                                                                          prob_mutation, prob_crossover,
                                                                                          tour_sel(3),
                                                                                          one_point_cross, muta_bin,
                                                                                          sel_survivors_elite(0.02),
                                                                                          merito(problem))

    return accumulated_generations, accumulated_diffs, crossover_probs, mutation_probs, phenotype, problem


def run_n_times(num_runs):
    """
    Executes N runs.
    In each run simulate for all ADs being tested.
    Return the results of all simulations of all runs.
    """
    size_items = NUM_ITEMS
    max_value = MAX_VALUE_ITEM
    corr_amplitude = CORR_AMPLITUDE

    accumulated_generations_array = [list(copy.deepcopy([])) for _ in range(NUM_ADS + 1)]
    accumulated_diffs_array = [list(copy.deepcopy([])) for _ in range(NUM_ADS + 1)]
    crossover_probs_array = [list(copy.deepcopy([])) for _ in range(NUM_ADS + 1)]
    mutation_probs_array = [list(copy.deepcopy([])) for _ in range(NUM_ADS + 1)]
    phenotype_array = [list(copy.deepcopy([])) for _ in range(NUM_ADS + 1)]
    problem_array = [list(copy.deepcopy([])) for _ in range(NUM_ADS + 1)]
    timing_array = [list(copy.deepcopy([])) for _ in range(NUM_ADS + 1)]

    print("-------------------------------------------------------")
    print("-------------------- ! Warning ! ----------------------")
    print("----- This might take a while, please take a seat -----")
    print("-------------------------------------------------------")

    for ith_run in range(1, num_runs + 1):
        # TODO: time the algorithms to see if there's a significant difference in performance
        print("\n\n---------- run " + str(ith_run))

        # generate a strongly correlated problem to be solved
        # (the greater the value, the greater the weight)
        problem = generate_strong_cor(size_items, max_value, corr_amplitude)

        # solve using our custom methods, "AD i"
        for ith_ad in range(NUM_ADS + 1):

            print("-------------------- AD" + str(ith_ad))

            start_time = time()
            accumulated_generations, accumulated_diffs, crossover_probs, mutation_probs, phenot, problem = simulate(
                ith_ad, problem, size_items)
            finish_time = time()

            accumulated_generations_array[ith_ad].append(accumulated_generations)
            accumulated_diffs_array[ith_ad].append(accumulated_diffs)
            crossover_probs_array[ith_ad].append(crossover_probs)
            mutation_probs_array[ith_ad].append(mutation_probs)
            phenotype_array[ith_ad].append(phenot)
            problem_array[ith_ad].append(problem)
            timing_array[ith_ad].append(finish_time - start_time)  # log elapsed time


    return [accumulated_generations_array, accumulated_diffs_array, crossover_probs_array, mutation_probs_array,
            phenotype_array, problem_array, timing_array]


"""
    If by any chance you are reading the comments this is the starting point for our algorithm.
    Enjoy your trip! But be warned, we're constantly _evolving_ our skills. Ha ha ha!
    Get it?! No? Ok. We'll show ourselves out...
"""
if __name__ == '__main__':

    seed(666)  # random number generation with fixed seed for reproduceable results

    # Problem specific
    NUM_ITEMS = 250  # 250 a 500
    MAX_VALUE_ITEM = 250  # 250 a 500
    CORR_AMPLITUDE = 10  # value = weight + amplitude, higher weight means higher value

    NUM_ADS = 5  # number of distinct custom appreaches employed besides standard
    NUMBER_OF_RUNS = 30  # TODO: 30 , statistically relevant ammount of runs

    # The usual EA parameters
    NUM_GENERATIONS = 1000  # 1000
    POPULATION_SIZE = 50  # 250?
    PROB_CROSSOVER = 0.80  # resposável por variações grandes no início
    PROB_MUTATION = 0.01  # resposável por variações pequenas no final

    # TODO: fine tune these
    # AD Approach specific parameters
    WINDOW_SIZE = 100  # number previous generations considered for altering the auto-adaptative parameters
    ACTIVATION_THRESHOLD = 10  # below this lower bound start reversing the crossover and mutation
    CONSECUTIVE_ACTIVATIONS = 5  # number of consecutive times that the threshhold must be surmounted for effect
    CROSSOVER_STEP = 0.01
    MUTATION_STEP = 0.001
    CROSSOVER_BOUND = 0.30  # lower bound for crossover prob.
    MUTATION_BOUND = 0.10  # upper bound for mutation prob.

    # location of saved files
    PATH = str(NUM_ITEMS) + "_" + str(MAX_VALUE_ITEM) + "_" + str(CORR_AMPLITUDE) \
        \
        + "_" + str(NUM_GENERATIONS) + "_" + str(POPULATION_SIZE) + "_" + str(PROB_CROSSOVER) + "_" + str(PROB_MUTATION) \
        \
        + "_" + str(WINDOW_SIZE) + "_" + str(ACTIVATION_THRESHOLD) + "_" + str(CONSECUTIVE_ACTIVATIONS) \
        + "_" + str(CROSSOVER_STEP) + "_" + str(MUTATION_STEP) + "_" + str(CROSSOVER_BOUND) + "_" + str(MUTATION_BOUND) \
        + "/"

    try:
        os.makedirs(PATH)
    except Exception as e:
        print(e)

    # run our simulations
    results = run_n_times(NUMBER_OF_RUNS)

    # analyse the results from the simulations
    analyse_results(PATH, NUMBER_OF_RUNS, NUM_ADS, results)

    # record the simulation's parameters
    dic = {
        "\n# Problem specific": " ---------- ",
        "Bag Capacity: ": NUM_ITEMS,
        "Max. item value: ": MAX_VALUE_ITEM,
        "Correlation amplitude: ": CORR_AMPLITUDE,
        "\n# The usual EA parameters": " ---------- ",
        "Num. generations: ": NUM_GENERATIONS,
        "Population size:: ": POPULATION_SIZE,
        "Prob. crossover: ": PROB_CROSSOVER,
        "Prob. mutation: ": PROB_MUTATION,
        "\n# AD Approach specific parameters": " ---------- ",
        "Refference window size: ": WINDOW_SIZE,
        "Activation threshold: ": ACTIVATION_THRESHOLD,
        "Consecutive activations: ": CONSECUTIVE_ACTIVATIONS,
        "Crossover step: ": CROSSOVER_STEP,
        "Mutation step: ": MUTATION_STEP,
        "Crossover lower bound: ": CROSSOVER_BOUND,
        "Mutation upper bound: ": MUTATION_BOUND
    }
    write_dic_to_file(PATH + "simulation_parameters.txt", dic)
