'''
Topic: Auto-Adaptation in Evolutionary Strategies
Authors: António Lima & Paulo Pereira
Remarks: Not for the faint of heart! Procede at your own risk.
'''


# criar a função run_n_times
# - tem todos os parâmetros que devem ser analisados estatisticamente
# --- os parâmetros, segundo o enunciado, terão a ver com um desvio padrão e uma média utilizados aquando da mutação e recombinação
# - executa a função run n vezes
# - guarda os resultados e parâmetros de execução num ficheiro
def run_n_times(num_runs):
	for i in range(num_runs):
		run()
    return


# criar a função run
# - executa uma simulação com os parâmetros fornecidos
# - devolve os resultados da experiência
def run():
    return


# criar a função de análise estatística e apresentação de gráficos
# - analisar os melhores resultados e os resultados da média
# - analisar o efeito das alterações nos parâmetros
def analyse():
    return


# starting pint for our algorithm
if __name__ == '__main__':