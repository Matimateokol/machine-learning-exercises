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
x = np.linspace(L_BOUND, U_BOUND, 100)
y = q(x)

np.random.seed(1)


#* F. Aktywacji - logistyczna sigmoidalna
def sigmoid(x):
    return 1/(1+np.exp(-x))

def leaky_relu(x):
    return 0.01*x

def d_leaky_relu(x):
    return 0.01

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
        self.LR = 0.3         #stała uczenia (do aktualizacji wag)

        #* Wagi z zakresu [0,1], bez biasow na razie
        self.W1 = np.random.randn(self.HIDDEN_L_SIZE, len(x)) #wagi miedzy warstwa wejsciowa a ukryta - rozmiar 9/10
        self.W2 = np.random.randn(len(y), self.HIDDEN_L_SIZE) #wagi miedzy warstwa ukryta a wyjsciowa - rozmiar 10/9

        #* biasy
        #self.B1 = np.random.randn(self.HIDDEN_L_SIZE)
        #self.B2 = np.random.randn(y.shape[0])

    #* Propagacja w przód
    def forward(self, x):

        #Warstwa pierwsza
        v = np.zeros(self.HIDDEN_L_SIZE)
        for i in range(0, self.W1.shape[0]-1):
            for j in range(0, len(x)-1):
                v[i] += x[j] * self.W1[i][j] #+ self.B1[i]

        y_hidden = np.zeros(len(v))
        for i in range(0, len(v)):
            y_hidden[i] = sigmoid(v[i])

        #Warstwa druga
        y_forward = np.zeros(len(y))
        for i in range(0, len(y)):
            for j in range(0, self.W2.shape[1] - 1):
                y_forward[i] += y_hidden[j] * self.W2[i][j] #+ self.B2[i]

        self.y_out = y_forward
        self.y_hidden = y_hidden

        return y_forward


    def predict(self, x):            
        return self.forward(x)
        
    def backward(self, x, y):

        # 1. Wczytanie pochodnej f. straty
        d_loss = d_nloss(self.y_out, y)

        # 2. Obliczanie gradientu dla warstwy wyjsciowej
        # 2.1 Gradient błędu w warstwie wyjściowej
        delta_output = d_loss * d_sigmoid(self.y_out)

        # 2.2 Gradient wag dla warstwy wyjściowej:
        grad_W2 = np.dot(delta_output.reshape(-1, 1), self.y_hidden.reshape(1, -1))
        #grad_B2 = delta_output

        # 3. Obliczenie gradientu dla warstwy ukrytej
        # 3.1 Błąd propagowany do warstwy ukrytej
        delta_hidden = np.dot(self.W2.T, delta_output) * d_sigmoid(self.y_hidden)

        # 3.2 Gradient wag dla warstwy ukrytej
        grad_W1 = np.dot(delta_hidden.reshape(-1, 1), x.reshape(1, -1))
        #grad_B1 = delta_hidden

        # 4. Aktualizacja wag
        self.W2 -= self.LR * grad_W2
        #self.B2 -= self.LR * grad_B2
        self.W1 -= self.LR * grad_W1
        #self.B1 -= self.LR * grad_B1
        
    def train(self, x_set, y_set, iters): #forward -> strata (opc) -> backward -> aktualizacja wag (w backward jest)
        for i in range(iters):
            self.forward(x_set)
            if i % 1000 == 0:
                loss = np.sum(nloss(self.y_out, y))
                print(f'Iteration {i}, Loss: {loss}')
            self.backward(x_set, y_set)

#* Inicjalizacja instacji sieci neuronowej i wytrenowanie jej (poczatkowo 15000 iteracji)        
nn = DlNet(x,y)

losses = nn.train(x, y, 15000)


yh = nn.predict(x) #ToDo tu umiescić wyniki (y) z sieci

import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(x,y, 'r')
plt.plot(x,yh, 'b')

plt.show()
