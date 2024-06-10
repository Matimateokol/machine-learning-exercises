#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:51:50 2021

@author: Rafał Biedrzycki
Kodu tego mogą używać moi studenci na ćwiczeniach z przedmiotu Wstęp do Sztucznej Inteligencji.
Kod ten powstał aby przyspieszyć i ułatwić pracę studentów, aby mogli skupić się na algorytmach sztucznej inteligencji. 
Kod nie jest wzorem dobrej jakości programowania w Pythonie, nie jest również wzorem programowania obiektowego, może zawierać błędy.

Nie ma obowiązku używania tego kodu.
"""

import numpy as np

#ToDo tu prosze podac pierwsze cyfry numerow indeksow
p = [3, 3]

L_BOUND = -5
U_BOUND = 5

#* Reprezentowana funkcja J(x)
def q(x):
    return np.sin(x*np.sqrt(p[0]+1))+np.cos(x*np.sqrt(p[1]+1))

#* J: [-5,5] → R
x = np.linspace(L_BOUND, U_BOUND, 10)
y = q(x)

np.random.seed(1)


#* F. Aktywacji - logistyczna sigmoidalna
def sigmoid(x):
    return 1/(1+np.exp(-x))

#* Pochodna F. Aktywacji
def d_sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s * (1-s)
     
#* F. straty (różnica kwadratowa między pożądanym wyjściem, a wyjściem sieci neuronowej)
def nloss(y_out, y):
    return (y_out - y) ** 2

#* Pochodna F. Straty
def d_nloss(y_out, y):
    return 2*( y_out - y )

#* Klasa od tworzenia sieci neuronowej    
class DlNet:

    #TODO: uzupełnić funkcje forward, backward, predict i train. Dodać wagi, na razie bez biasu

    #! Czy w tej funkcji nie trzeba czegoś zwracać?
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.y_out = 0
        
        self.HIDDEN_L_SIZE = 9  #liczba neuronów w warstwie ukrytej
        self.LR = 0.003         #stała uczenia (do aktualizacji wag)

        #* Wagi z zakresu [0,1], bez biasow na razie
        #? Dla mnie powinno być tak jak nizej wagi macierzami, wedlug gpt 1*self.HIDDEN_L_SIZE huj wi dlaczego
        self.W1 = np.random.randn(self.HIDDEN_L_SIZE, len(x)) #wagi miedzy warstwa wejsciowa a ukryta - rozmiar 9/10
        self.W2 = np.random.randn(len(y), self.HIDDEN_L_SIZE) #wagi miedzy warstwa ukryta a wyjsciowa - rozmiar 10/9

        #print(self.W1[0][9])
        #print(self.W2[9][8])

    #* Propagacja w przód
    def forward(self, x):

        #Warstwa pierwsza
        v = np.zeros(self.HIDDEN_L_SIZE)
        for i in range(0, self.W1.shape[1]-1):
            for j in range(0, len(x)-1):
                v[i] += x[j] * self.W1[i][j]

        y_hidden = np.zeros(len(v))
        for i in range(0, len(v)):
            y_hidden[i] = sigmoid(v[i])

        #Warstwa druga
        v_2nd = np.zeros(len(y))
        for i in range(0, len(y)):
            for j in range(0, self.W2.shape[1] - 1):
                v_2nd[i] += y_hidden[j] * self.W2[i][j]

        y_forward = np.zeros(len(v_2nd))
        for i in range(0, len(v_2nd)):
            y_forward[i] = sigmoid(v_2nd[i])

        return y_forward


    def predict(self, x):            
        return self.forward(x)
        
    def backward(self, x, y):
        pass
        
    def train(self, x_set, y_set, iters):    
        for i in range(0, iters):
            pass

#* Inicjalizacja instacji sieci neuronowej i wytrenowanie jej (poczatkowo 15000 iteracji)        
nn = DlNet(x,y)

nn.train(x, y, 15000)

yh = [] #ToDo tu umiescić wyniki (y) z sieci

import matplotlib.pyplot as plt


# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

# plt.plot(x,y, 'r')
# plt.plot(x,yh, 'b')

# plt.show()
