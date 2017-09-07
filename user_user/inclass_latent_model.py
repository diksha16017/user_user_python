import numpy as np
from copy import deepcopy


Y = np.random.randint(low=0, high=5, size=(4, 3))
print 'original matrix'
print Y
print '*****'
R = np.zeros((4, 3))
XO = deepcopy(Y)
count = 21
k = 2

def initialize():
    global XO
    global R
    global Y
    for i in range(len(XO)):
        for j in range(len(XO[0])):
            if XO[i][j] == 0:
                row = np.mean(XO[i, :])
                column = np.mean(XO[:, j])
                mean_ = float((row + column)/2.0)
                XO[i][j] = mean_
            else:
                R[i][j] = 1

    print 'XO matrix'
    print XO
    print '************'
    print 'R matrix'
    print R
    print '***********'

def latent_method():

    global count
    global XO
    global R
    global Y
    global k
    print 'latent method'
    while count > 0:
        count = count - 1
        B = XO + (Y - np.multiply(R, XO))
        UK, VK = matrix_factorisation(B)
        XO = np.dot(UK, VK)
    print 'final UK'
    print UK
    print 'final VK'
    print VK
    print 'final Ratings matrix '
    print np.dot(UK, VK)

def matrix_factorisation(B):
    count1 = 20
    global k
    lamb_reg = [0.2] * k
    U = np.random.randint(low=1, high=5, size=(4, k))
    while count1 > 0:
        count1 = count1 - 1
        U_trans = np.transpose(U)
        VK = np.linalg.lstsq(np.dot(U_trans, U) + np.diag(lamb_reg), np.dot(np.transpose(U), B))[0]
        VK_trans = np.transpose(VK)
        R_trans = np.transpose(B)
        UK_temp = np.linalg.lstsq(np.dot(VK, VK_trans) + np.diag(lamb_reg), np.dot(VK, R_trans))[0]
        UK = np.transpose(UK_temp)
        U = UK
        if np.array_equal(np.dot(UK, VK), B) == True:
            break
    return UK, VK



initialize()
latent_method()
