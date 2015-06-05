__author__ = 'paulo'

from plot_auto_adapt import read_data_from_file
from numpy import mean, std
from stat_2015_alunos import histogram, box_plot, histogram_norm, one_way_ind_anova, t_test_ind

def analyse_p_value(data, text, p_prob=0.05):
    p_value = one_way_ind_anova(data)[1]
    print("P-Value " + text + ": ", p_value)

    if p_value < p_prob:
        for i in range(len(data)):
            for y in range(i+1, len(data)):
                print("\t"+str(i)+":"+str(y)+" ", t_test_ind(data[i], data[y])[1])




if __name__ == '__main__':
    # Problem specific
    NUM_ITEMS = 250  # 250 - 500
    MAX_VALUE_ITEM = 250  # 250 - 500
    CORR_AMPLITUDE = MAX_VALUE_ITEM / 10  # value = weight + amplitude, higher weight means higher value
    NUM_ADS = 5  # number of distinct custom appreaches employed besides standard
    NUMBER_OF_RUNS = 30  # 30 | statistically relevant amount fo results
    # The usual EA parameters
    NUM_GENERATIONS = 1000  # 800+ | less than 800 may prove insufficient for some ADs
    POPULATION_SIZE = 50  # 50 - 250 | number of individuals in population
    PROB_CROSSOVER = 0.80  # 0.75 - 0.85 | responsible for the fast initial convergence
    PROB_MUTATION = 0.01  # 0.01 - 0.10 | responsible for the small variations in later geneerations
    # AD Approach specific parameters
    WINDOW_SIZE = 100  # 25 - 100 | number previous generations considered for altering the auto-adaptative parameters
    ACTIVATION_THRESHOLD = 10  # 0 - 10 | below this comparison lower bound change the crossover and mutation
    CONSECUTIVE_ACTIVATIONS = 5  # 3 - 7 | number of consecutive times that the threshhold must be surmounted for effect
    # Our additional EA parameters
    CROSSOVER_STEP = 0.01  # how much the crossover prob. decreases after activation
    MUTATION_STEP = 0.001  # how much the crossover prob. increases after activation
    CROSSOVER_BOUND = 0.30  # lower bound for crossover prob.
    MUTATION_BOUND = 0.10  # upper bound for mutation prob.
    # location of saved files
    PATH = str(NUM_ITEMS) + "_" + str(MAX_VALUE_ITEM) + "_" + str(CORR_AMPLITUDE) \
 \
           + "_" + str(NUM_GENERATIONS) + "_" + str(POPULATION_SIZE) + "_" + str(PROB_CROSSOVER) + "_" + str(
        PROB_MUTATION) \
 \
           + "_" + str(WINDOW_SIZE) + "_" + str(ACTIVATION_THRESHOLD) + "_" + str(CONSECUTIVE_ACTIVATIONS) \
           + "_" + str(CROSSOVER_STEP) + "_" + str(MUTATION_STEP) + "_" + str(CROSSOVER_BOUND) + "_" + str(
        MUTATION_BOUND)

    # statistics

    data_mean = [list() for i in range(NUM_ADS + 1)]
    data_max = [list() for i in range(NUM_ADS + 1)]

    # buscar o best da ultima interação de cada run
    for run_i in range(1, NUMBER_OF_RUNS + 1):
        for ad_i in range(NUM_ADS + 1):
            accumulated_generations = read_data_from_file(PATH + "/run_" + str(run_i) + "/data_AD" + str(ad_i) + "/generations.json")
            data_mean[ad_i].append(mean(accumulated_generations[-1]))
            data_max[ad_i].append(max(accumulated_generations[-1]))

    # obter a média para as X runs
    runs_median = list()
    runs_median_std = list()
    runs_max = list()
    runs_max_std = list()
    for ad_i in range(NUM_ADS + 1):
        runs_median.append(mean(data_mean[ad_i]))
        runs_median_std.append(std(data_mean[ad_i]))
        runs_max.append(mean(data_max[ad_i]))
        runs_max_std.append(std(data_max[ad_i]))

        # histogram_norm(data_mean[ad_i], "Histogram for the maximums values in AD" + str(ad_i), "Values", "Frequency")
        # histogram_norm(data_max[ad_i], "Histogram for the maximums values in AD" + str(ad_i), "Values", "Frequency")

    analyse_p_value(data_mean, "mean")
    analyse_p_value(data_max, "max")

    box_plot(data_mean, [str(i) for i in range(len(data_mean))])
    box_plot(data_max, [str(i) for i in range(len(data_max))])

