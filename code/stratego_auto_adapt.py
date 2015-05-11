'''
Topic: Auto-Adaptation in Evolutionary Strategies
Authors: António Lima & Paulo Pereira
Remarks: Not for the faint of heart! Procede at your own risk.
'''

from sea_bin import *
from kp_1 import *
from random import *


# tendo em conta a janela de referência, decidir se devem ser alterados os parâmetros
# - tamanho da população
# - probabilidade de crossover
# - probabilidade de mutação
def auto_adapt_fitness():
    return


# criar a função run
# - executa uma simulação com os parâmetros fornecidos
# - devolve os resultados da experiência
def run(auto_adapt, refference_window, population_size, prob_crossover, prob_mutation):

    problem = generate_strong_cor(10, 10, 5)
    fit = merito(problem)
    best = sea(100, 50, 10, 0.1, 0.8, tour_sel(3), one_point_cross, muta_bin, sel_survivors_elite(0.02), fit)
    display(best, phenotype, problem)

    return


# criar a função run_n_times
# - tem todos os parâmetros que devem ser analisados estatisticamente
# --- os parâmetros, segundo o enunciado, terão a ver com um desvio padrão e uma média utilizados
# --- aquando da mutação e recombinação
# - executa a função run n vezes
# - guarda os resultados e parâmetros de execução num ficheiro
def run_n_times(num_runs, refference_window, population_size, prob_crossover, prob_mutation):
    results_with_auto_adapt = []
    results_without_auto_adapt = []

    for i in range(num_runs):
        results_with_auto_adapt.append(run(True, refference_window, population_size, prob_crossover, prob_mutation))
        results_without_auto_adapt.append(run(False, refference_window, population_size, prob_crossover, prob_mutation))


    return results_with_auto_adapt, results_without_auto_adapt


# criar a função de análise estatística e apresentação de gráficos
# - analisar os melhores resultados e os resultados da média
# - analisar o efeito das alterações nos parâmetros
def analyse(data):
    return


# starting point for our algorithm
if __name__ == '__main__':

    seed(666)  # random number generation with fixed seed for reproduceable results

    number_of_runs = 1  # TODO: 30  # statistically relevant ammount of runs

    refference_window = 1  # number previous generations to eb considered for altering the auto-adaptative parameters

    population_size = 250  # the number of elements influences the effect fo reproduction because the gene pool is bigger
    prob_crossover = 0.70  # resposável por salto grande no início
    prob_mutation = 0.01  # resposável por saltos pequenos no final

    results_with_auto_adapt, results_without_auto_adapt = run_n_times(number_of_runs, refference_window,
                                                                      population_size, prob_crossover, prob_mutation)

    analyse(results_with_auto_adapt)
    analyse(results_without_auto_adapt)








