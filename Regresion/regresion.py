# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression():

    def _init_(self) -> None:
        pass

    def fit(self, X, Y, learning_rate=0.0000001, epochs=100000, bias=True):
        n = int(len(X))  # numero de elementos de x
        y = Y.reshape(n, 1)  # convertimos y en un vector columna
        if bias:
            m = X.shape[1] + 1
            aux = np.ones((n, 1))
            X = np.concatenate((X, aux), axis=1)
        else:
            m = X.shape[1]
        thetas = np.zeros((m, 1))

        errores = []
        iteraciones = []

        # regresion logistica
        for i in range(epochs):
            h = 1 / (1 + np.exp(-np.dot(X, thetas)))
           
            error = h - y
            gradient = np.dot(X.T, error) / n
            cost = np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h)) / n #Esta es la funcion de costo
            thetas = thetas - learning_rate * gradient
            errores.append(cost)
            iteraciones.append(i)
        print (thetas)     
        return iteraciones, errores
   
    
       