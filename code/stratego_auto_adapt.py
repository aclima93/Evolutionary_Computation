"""
Topic: Auto-Adaptation in Evolutionary Strategies
Authors: António Lima & Paulo Pereira
Remarks: Our proposal of an Auto-Adapting Evolutionary Algorithm
"""

from random import *
from kp_1 import *
from analyse import *


def debug_print(something):
    # print to console
    if DEBUG:
        print(something)

    # store in log file
    if LOG_OUTPUT:
        f.write(str(something))
        f.write("\n")

    return


def average_reff_pop(refference_window, fitness_func):
    """
    returns the average population based on the ith average individual of all populations
    """
    num_populations = len(refference_window)

    if num_populations > 1:
        average_population = []
        ith_pairings = list(zip(*refference_window))
        for ith_pair in ith_pairings:
            average_population.append(average_indiv(ith_pair, fitness_func))
        return average_population

    else:
        return refference_window[0]


def average_indiv(population, fitness_func):
    """
    returns the average individual based on all individuals in the population
    """
    individuals = []
    for indiv, fit in population:
        individuals.append(indiv)

    num_individuals = len(individuals)
    average_individual = [0] * len(individuals[0])  # individual with only zeros
    # sum all the solutions
    for indiv in individuals:
        for i in range(len(indiv)):
            average_individual[i] += indiv[i]

    # divide by number of individuals in population
    for i in range(len(average_individual)):
        average_individual[i] = round(average_individual[i] / num_individuals)

    return [average_individual, fitness_func(average_individual)]


def compare_individs(individual1, individual2):
    """
    compares individuals by differences in genotype
    """
    diffs = 0
    for i in range(len(individual1)):
        if individual1[i] != individual2[i]:
            diffs += 1
    return diffs


def auto_adapt_fitness(cur_population, refference_window, fitness_func):
    """
    Analyses the populations stored in the refference window and creates an average population.
    From this average population we derive an average individual.
    Then we also derive the average individual for the current population
    and return the number of differences between them.
    """

    """
    debug_print("\n\n----------------------------------")
    debug_print("cur_population: ")
    debug_print(cur_population)

    debug_print("refference_window: ")
    debug_print(refference_window)
    """

    # get average population from refference_window
    average_reff_population = average_reff_pop(refference_window, fitness_func)

    """
    debug_print("average_reff_population: ")
    debug_print(average_reff_population)
    """

    # get average individual from average population
    average_reff_individual = average_indiv(average_reff_population, fitness_func)

    debug_print("average_reff_individual: ")
    debug_print(average_reff_individual)

    # get average individual from current population
    average_individual = average_indiv(cur_population, fitness_func)

    debug_print("average_individual: ")
    debug_print(average_individual)

    # compare average individuals
    differences = compare_individs(average_reff_individual, average_individual)

    debug_print("num differences: ")
    debug_print(differences)

    return differences


def stratego_next_population(elite):
    def elitism(parents, offspring):
        """
        Survivals: elitism
        tendo em conta a janela de referência, alterar o tamanho da população
        """
        # TODO: yeah... are we still going to do this?
        size = len(parents)
        comp_elite = int(size * elite)
        offspring.sort(key=itemgetter(1), reverse=True)
        parents.sort(key=itemgetter(1), reverse=True)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]

        return new_population

    return elitism


def stratego(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination, mutation,
             sel_survivors, fitness_func, refference_window_size, refference_window):
    """
    funcao stratego
    - variação da sea fornecida pelo prof. Ernesto mas que aplica os conceitos que estamos a estudar:
    --- variação do tamanho da população, prob. de mutação e prob. de crossover
    returns the best individual in the final generation as well as all generations and the list of all differences
    """

    accumulated_generations = []
    accumulated_differences = []
    crossover_probs = []
    mutation_probs = []
    difference_history = 0

    # inicializa população: indiv = (cromo,fit)
    populacao = gera_pop(size_pop, size_cromo)
    accumulated_generations.append(populacao)

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

        for indiv, fit in progenitores:
            novo_indiv = mutation(indiv, prob_mut)
            descendentes.append((novo_indiv, fitness_func(novo_indiv)))

        # added previous generation to refference_window and remove the oldest (or just append until we have enough)
        if len(refference_window) == refference_window_size:
            refference_window.remove(refference_window[0])
        refference_window.append(populacao)

        # New population
        populacao = sel_survivors(populacao, descendentes)
        # Avalia nova _população
        populacao = [(indiv[0], fitness_func(indiv[0])) for indiv in populacao]

        # store the population
        accumulated_generations.append(populacao)

        # add or remove individuals to new population based on the number of differences found
        num_differences = auto_adapt_fitness(populacao, refference_window, fitness_func)
        accumulated_differences.append(num_differences)
        ratio = num_differences / len(populacao[0][0])

        debug_print("ratio: ")
        debug_print(ratio)

        # if the difference ratio falls below the threshhold alter the crossover and mutation probabilities
        if ratio < ACTIVATION_THRESHOLD:
            difference_history += 1

        if difference_history == DIFFERENCE_TOLERANCE:

            # reset the counter
            difference_history = 0

            if (prob_cross - CROSSOVER_STEP) > CROSSOVER_BOUND:
                prob_cross -= CROSSOVER_STEP

            if (prob_mut + MUTATION_STEP) < MUTATION_BOUND:
                prob_mut += MUTATION_STEP

        crossover_probs.append(prob_cross)
        mutation_probs.append(prob_mut)

    return [accumulated_generations, accumulated_differences, crossover_probs, mutation_probs]


def simulate(auto_adapt, problem, size_items):
    """
    função simulate
    - executa uma simulação com os parâmetros fornecidos
    - devolve os resultados da experiência
    """

    refference_window_size = WINDOW_SIZE
    refference_window = []

    num_generations = NUM_GENERATIONS
    population_size = POPULATION_SIZE
    prob_crossover = PROB_CROSSOVER
    prob_mutation = PROB_MUTATION

    if auto_adapt:
        # stratego(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination,
        # mutation, sel_survivors, fitness_func, refference_window_size, refference_window)
        sim_data = stratego(num_generations, population_size, size_items, prob_mutation, prob_crossover, tour_sel(3),
                            one_point_cross, muta_bin, stratego_next_population(0.02), merito(problem),
                            refference_window_size, refference_window)
    else:
        # sea(numb_generations, size_pop, size_cromo, prob_mut, prob_cross, sel_parents, recombination,
        # mutation, sel_survivors, fitness_func)
        sim_data = sea(num_generations, population_size, size_items, prob_mutation, prob_crossover, tour_sel(3),
                       one_point_cross, muta_bin, sel_survivors_elite(0.02), merito(problem))

    return [sim_data, phenotype, problem]


def run_n_times(num_runs):
    """
    funcao run_n_times
    - tem todos os parâmetros que devem ser analisados estatisticamente
    --- os parâmetros, segundo o enunciado, terão a ver com um desvio padrão e uma média utilizados
    --- aquando da mutação e recombinação
    - executa a função simulate n vezes
    - guarda os resultados e parâmetros de execução num ficheiro
    """
    size_items = NUM_ITEMS
    max_value = MAX_VALUE_ITEM

    results_with_auto_adapt = []
    results_without_auto_adapt = []

    for ith_run in range(1, num_runs + 1):

        # TODO: time the algorithms to see if there's a significant difference in performance
        print("Run Number " + str(ith_run))

        # generate a problem to be solved
        problem = generate_uncor(size_items, max_value)

        # solve using our custom method, "stratego"
        results_with_auto_adapt.append(simulate(True, problem, size_items))

        # solve the "traditional way"
        results_without_auto_adapt.append(simulate(False, problem, size_items))

    return [results_with_auto_adapt, results_without_auto_adapt]


"""
If by any chance you are reading the comments this is the starting point for our algorithm.
Enjoy your trip! But be warned, we're constantly _evolving_ our skills. Ha ha ha!
Get it?! No? Ok. We'll show ourselves out...
"""
if __name__ == '__main__':
    f = open('output.txt', 'w')
    seed(666)  # random number generation with fixed seed for reproduceable results

    DEBUG = False
    LOG_OUTPUT = False

    # Problem specific
    NUM_ITEMS = 50  # 50
    MAX_VALUE_ITEM = 50  # 50

    # The usual EA parameters
    NUMBER_OF_RUNS = 5  # TODO: 30 , statistically relevant ammount of runs
    NUM_GENERATIONS = 500  # 500
    POPULATION_SIZE = 250  # 250
    PROB_CROSSOVER = 0.80  # resposável por variações grandes no início
    PROB_MUTATION = 0.10  # resposável por variações pequenas no final

    # AD Approach specific parameters
    WINDOW_SIZE = 10  # number previous generations considered for altering the auto-adaptative parameters
    ACTIVATION_THRESHOLD = 0.25  # below this lower bound start reversing the crossover and mutation
    DIFFERENCE_TOLERANCE = 10  # number of times that the threshhold must be surmounted before we take action (give the algorithm time to sort itself out)
    CROSSOVER_STEP = 0.10
    MUTATION_STEP = 0.10
    CROSSOVER_BOUND = 0.10
    MUTATION_BOUND = 0.80

    PATH = "results/"

    # record the simulation's parameters
    dic = {
        "\n# Problem specific": " ---------- ",
        "Bag Capacity: ": NUM_ITEMS,
        "Max. item value: ": MAX_VALUE_ITEM,
        "\n# The usual EA parameters": " ---------- ",
        "Num. generations: ": NUM_GENERATIONS,
        "Population size:: ": POPULATION_SIZE,
        "Prob. crossover: ": PROB_CROSSOVER,
        "Prob. mutation: ": PROB_MUTATION,
        "\n# AD Approach specific parameters": " ---------- ",
        "Refference window size: ": WINDOW_SIZE,
        "Activation threshold: ": ACTIVATION_THRESHOLD,
        "Difference tolerance: ": DIFFERENCE_TOLERANCE,
        "Crossover step: ": CROSSOVER_STEP,
        "Mutation step: ": MUTATION_STEP,
        "Crossover lower bound: ": CROSSOVER_BOUND,
        "Mutation upper bound: ": MUTATION_BOUND
    }
    write_dic_to_file(PATH + "simulation_parameters.txt", dic)

    # run our simulations
    results = run_n_times(NUMBER_OF_RUNS)

    # analyse the results from the simulations
    analyse_results(PATH, NUMBER_OF_RUNS, results)
