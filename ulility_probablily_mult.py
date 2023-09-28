import numpy as np
import matplotlib.pyplot as plt



prob_matrix = np.array([[0.008,0.096,0.384,0.323],[0.008,0.096,0.384,0.323]])

utility_matrix =1000*np.array([[909,0.096,0.384,0.323],[7,0.096,0.384,0.8]])


def mult_probability_and_utility(prob_matrix,utility_matrix):
    utility_prob_matrice = prob_matrix*utility_matrix
    summer_matrix = []
    for i in range(0,len(utility_prob_matrice[1][:])):
        a = sum(utility_prob_matrice[:][i])
        summer_matrix.append(a)
    summer_matrix = np.array(summer_matrix).transpose()
    return summer_matrix, utility_prob_matrice


utility_prob_matrice_sum, utility_prob_matrice = mult_probability_and_utility(prob_matrix,utility_matrix)
print(utility_prob_matrice_sum)
print(utility_prob_matrice)