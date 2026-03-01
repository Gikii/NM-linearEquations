import timeit

import numpy as np
import matplotlib.pyplot as plt

#https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html
def jacobi(A, b):
    N = len(b)
    x = np.zeros(N)
    residumLimit = 1e-9
    maxIterations = 1000

    D = np.diag(A)
    R = A - np.diagflat(D)

    residuums = []
    time0 = timeit.default_timer()

    for _ in range(maxIterations):
        x_new = (b - np.dot(R, x)) / D


        residuum = b - np.dot(A, x_new)
        residuums.append(np.linalg.norm(residuum, ord=2))

        if residuums[-1] < residumLimit:
            break

        x = x_new
    time1 = timeit.default_timer()
    deltaTime = time1 - time0
    return x_new, residuums, deltaTime

def gaussSeidel(A,b):
    N = len(b)
    x = np.zeros(N)
    residumLimit = 1e-9
    maxIterations = 1000

    residuums = []
    time0 = timeit.default_timer()

    for _ in range(maxIterations):
        x_old = x.copy()

        for i in range(N):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])

            x[i] = (b[i] - sum1 - sum2) / A[i, i]


        residuum = np.dot(A, x) - b

        resid_norm = np.linalg.norm(residuum, ord=2)
        residuums.append(resid_norm)
        if resid_norm < residumLimit:
            break

    time1 = timeit.default_timer()
    deltaTime = time1 - time0
    return x, residuums, deltaTime

def matrixEquation(N, a1, a2, a3):
    A = np.zeros((N, N))

    np.fill_diagonal(A, a1)  # a1
    np.fill_diagonal(A[:, 1:], a2)  # a2
    np.fill_diagonal(A[:, 2:], a3)  # a3
    np.fill_diagonal(A[1:, :], a2)  # a2
    np.fill_diagonal(A[2:, :], a3)  # a3
    return A

def plotJacobiwithGauss(residuums_jac, residuums_gauss):
    plt.plot(range(len(residuums_jac)), residuums_jac, label='Jacobi')
    plt.plot(range(len(residuums_gauss)), residuums_gauss, label='Gauss-Seidel')
    plt.yscale('log')
    plt.xlabel('Iteracje')
    plt.ylabel('Norma residuum')
    plt.title('Zmiana normy residuum w kolejnych iteracjach')
    plt.legend()
    plt.grid()
    plt.show()

def LUfactorization(N,A):
    L = np.eye(N)
    U = A.copy()

    for i in range(N):
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]

    return L, U

def forward_substitution(L, b):
    N = len(b)
    y = np.zeros(N)

    for i in range(N):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y

def backward_substitution(U, y):
    N = len(y)
    x = np.zeros(N)

    for i in range(N-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

def LUdecomposition(N, matrix):
    time0 = timeit.default_timer()
    L, U = LUfactorization(N, matrix)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    time1 = timeit.default_timer()
    deltaTime = time1 - time0
    return x, deltaTime


#Numer indeksu 198254
N = 1254
e = 2
#N = 15
f = 8
A = matrixEquation(N, 5+e, -1, -1)

b = np.sin(np.arange(1, N + 1) * (f + 1))

print("\n===ZADANIE B===")
print("     ===JACOBI===")
x_jac, res_jac, dTime = jacobi(A,b)
#print(x_jac)
print("Time:", round(dTime*1000,2), "ms")
print("Iterations:", len(res_jac))


print("     ===GAUSS===")
x_gauss, res_gauss, dTime = gaussSeidel(A,b)
#print(x_gauss)
print("Time:", round(dTime*1000,2), "ms")
print("Iterations:", len(res_gauss))
#print(res_gauss)

plotJacobiwithGauss(res_jac,res_gauss)


#ZADANIE C:
print("\n===ZADANIE C===")
C = matrixEquation(N, 3, -1, -1)

print("     ===JACOBI===")
xC_jac, resC_jac, dTime = jacobi(C,b)
#print(xC_jac)
print("Time:", round(dTime,2), "s")
print("Iterations:", len(resC_jac))

print("     ===GAUSS===")
xC_gauss, resC_gauss, dTime = gaussSeidel(C,b)
# print(xC_gauss)
print("Time:", round(dTime,2), "s")
print("Iterations:", len(resC_gauss))

plotJacobiwithGauss(resC_jac, resC_gauss)

#ZADANIE D:
print("\n===ZADANIE D===")

x, dTime = LUdecomposition(N, C)
#print(x)
print("Time:", round(dTime,2), "s")

residuum = np.dot(C, x) - b
resid_norm = np.linalg.norm(residuum, ord=2)
print("Residum norm:", resid_norm)


#ZADANIE E:
print("\n===ZADANIE E===")
arr = (100, 500, 1000, 1500, 2000, 2500, 3000)
jacobiTime = []
gaussTime = []
luTime = []
for n in arr:
    A = matrixEquation(n, 5+e, -1, -1)
    b = np.sin(np.arange(1, n + 1) * (f + 1))
    x_jac, res_jac, jTime = jacobi(A,b)
    jacobiTime.append(jTime)
    x_gauss, res_gauss, gTime = gaussSeidel(A, b)
    gaussTime.append(gTime)
    x_lu, lTime = LUdecomposition(n,A)
    luTime.append(lTime)

print("Time for Jacobi method[ms]:           ", [round(val*1000, 2) for val in jacobiTime])
print("Time for Gauss-Seidel method[ms]:     ", [round(val*1000, 2) for val in gaussTime])
print("Time for LU decomposition method[ms]: ", [round(val*1000, 2) for val in luTime])


plt.figure(figsize=(12, 5))

plt.plot(arr, jacobiTime, marker='o', label='Jacobi')
plt.plot(arr, gaussTime, marker='o', label='Gauss-Seidel')
plt.plot(arr, luTime,marker='o', label='LU Decomposition')
plt.xlabel("Liczba niewiadomych N")
plt.ylabel("Czas [s]")
plt.title("Porównanie metod - skala liniowa")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))

plt.plot(arr, jacobiTime, marker='o', label='Jacobi')
plt.plot(arr, gaussTime, marker='o', label='Gauss-Seidel')
plt.plot(arr, luTime, marker='o', label='LU Decomposition')
plt.xlabel("Liczba niewiadomych N")
plt.ylabel("Czas [s] (log)")
plt.yscale("log")
plt.title("Porównanie metod - skala logarytmiczna")
plt.legend()
plt.grid(True)
plt.show()




