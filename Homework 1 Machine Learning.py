#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:55:35 2025

@author: joshkevil
Homework 1- Introduction to Machine Learning
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
#---------------------------------PROBLEM 1--------------------------------------------------------------------
#define vectors
x = np.array([3.4, 1.5, 0, -2.2])
y = np.array([1.3, 4.2, 3.0, 2.5])

#define n- norm based on definition
def nnorm(x,n):
    SUM = 0
    for num in x:
        SUM+= num**n
    return SUM**(1/n)

#calculate norms        
L1 = np.linalg.norm(x,ord = 1)
L2 = np.linalg.norm(x,ord = 2)
L3 = nnorm(x,3)
L4 = np.linalg.norm(x,ord = np.infty)

#put norms in dictionary
norms = {"L1":L1,"L2":L2,"L3":L3,"L4":L4}

#sort dictionary
sorted_dict = dict(sorted(norms.items(), key=lambda item: item[1]))

# Print 
print(sorted_dict)

#subtract vectors 
z = x-y

#calculate distances with norms
dL1 = np.linalg.norm(z,ord = 1)
dL2 = np.linalg.norm(z,ord = 2)
dL4 = np.linalg.norm(z,ord = np.infty)

#format into dictionary for sorting
norms = {"dL1":dL1,"dL2":dL2,"dL4":dL4}

#sort
sorted_dict_dist = dict(sorted(norms.items(), key=lambda item: item[1], reverse = True))

#Print
print(sorted_dict_dist)

#---------------------------------PROBLEM 2---------------------------------------------------------------
# defie matrix
B = np.array([[2, 2, 4], [1, 4, 7], [2, 5, 4]])

# fidn eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(B)

# diagonalize eigenvalues
Lambda = np.diag(eigenvalues)

# augment eigenvector matrix
Q = eigenvectors

# verify decomposition
Q_inv = np.linalg.inv(Q)
reconstructed_B = Q @ Lambda @ Q_inv  

# Print results
print("Matrix B:")
print(B)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors (columns of Q):")
print(Q)
print("\nDiagonal matrix Λ:")
print(Lambda)
print("\nReconstructed B (Q Λ Q^(-1)):")
print(reconstructed_B)

# Check if the reconstruction is close to the original B (accounting for floating-point errors)
print("\nIs B equal to Q Λ Q^(-1)?", np.allclose(B, reconstructed_B))

#---------------------------------PROBLEM 3-------------------------------------------------------------------

#A): 


# Define symbolic variables
x, y = sp.symbols('x y')

# Define the function
f = x**2 + 2*y**2 - 2*x*y

# Compute the Hessian matrix
hessian = sp.Matrix([[sp.diff(f, var1, var2) for var1 in (x, y)] for var2 in (x, y)])

print("Hessian matrix:")
print(hessian)


#D)
#dDef function 
def f(x, y):
    return x**2 + 2*y**2 - 2*x*y

# def the gradient funct
def gradient_f(x, y):
    df_dx = 2*x - 2*y 
    df_dy = 4*y -2*x  
    return np.array([df_dx, df_dy])

# gradient Descent Algorithm
def gradient_descent(starting_point, learning_rate, num_iterations):
    # initialize guess at starting point
    w = np.array(starting_point, dtype=float)
    
    history = {'x': [], 'y': [], 'f': []}
    
    for i in range(num_iterations):
        #calculate grad
        grad = gradient_f(w[0], w[1])
        
        # update the parameters 
        w = w - learning_rate * grad

        history['x'].append(w[0])
        history['y'].append(w[1])
        history['f'].append(f(w[0], w[1]))
        
    
    return w, history

# Parameters
starting_point = [5, 10]  # guess
learning_rate = 0.001       # learning rate
num_iterations = 50000       # iterations

final_point, history = gradient_descent(starting_point, learning_rate, num_iterations)

print("\nFinal result:")
print(f"Optimal point: [x, y] = {final_point}")
print(f"Minimum value of f(x, y): {f(final_point[0], final_point[1])}")

# Plot the convergence curves
plt.figure(figsize=(15, 5))

#f(x, y) vs iterations
plt.subplot(1, 3, 1)
plt.plot(range(num_iterations), history['f'], marker='o', color='b')
plt.xlabel('Iterations')
plt.ylabel('f(x, y)')
plt.title('f(x, y) vs Iterations')
plt.grid()

# x vs iterations
plt.subplot(1, 3, 2)
plt.plot(range(num_iterations), history['x'], marker='o', color='r')
plt.xlabel('Iterations')
plt.ylabel('x')
plt.title('x vs Iterations')
plt.grid()

# y vs iterations
plt.subplot(1, 3, 3)
plt.plot(range(num_iterations), history['y'], marker='o', color='g')
plt.xlabel('Iterations')
plt.ylabel('y')
plt.title('y vs Iterations')
plt.grid()

plt.tight_layout()
plt.show()
    

