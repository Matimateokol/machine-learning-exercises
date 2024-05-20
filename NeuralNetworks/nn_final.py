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

#Aleksander Bujnowski, Mateusz Kołacz

import numpy as np

#Pierwsze cyfry numerow indeksow
p = [5, 0]

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

    # CO ZROBIONO: uzupełniono funkcje forward, backward, predict i train. Dodano wagi, na razie bez biasu

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.y_out = 0
        
        self.HIDDEN_L_SIZE = 9  #liczba neuronów w warstwie ukrytej
        self.LR = 0.003        #stała uczenia (do aktualizacji wag)
        self.QUALITY_INDICATOR = 0 #zmienna do przechowania wskaznika jakosci ostatniej aproksymacji

        #* Wagi z zakresu [0,1)
        self.W_in_hidden = np.random.randn(self.HIDDEN_L_SIZE, len(x)) #wagi miedzy warstwa wejsciowa a ukryta - rozmiar 9/10
        self.W_hidden_out = np.random.randn(len(y), self.HIDDEN_L_SIZE) #wagi miedzy warstwa ukryta a wyjsciowa - rozmiar 10/9

    #* Propagacja w przód
    def forward(self, x):

        #Warstwa pierwsza
        v = np.zeros(self.HIDDEN_L_SIZE)
        for i in range(0, self.W_in_hidden.shape[0]-1):
            for j in range(0, len(x)-1):
                v[i] += x[j] * self.W_in_hidden[i][j] 

        y_hidden = np.zeros(len(v))
        for i in range(0, len(v)):
            y_hidden[i] = sigmoid(v[i])

        #Warstwa druga
        y_forward = np.zeros(len(y))
        for i in range(0, len(y)):
            for j in range(0, self.W_hidden_out.shape[1] - 1):
                y_forward[i] += y_hidden[j] * self.W_hidden_out[i][j] 

        self.y_out = y_forward
        self.y_hidden = y_hidden

        #Zwrot wyników z drugiej warstwy
        return self.y_out

    def predict(self, x):            
        return self.forward(x)
        
    #* Propagacja w tył
    def backward(self, x, y):

        # 1. Wczytanie pochodnej f. straty
        dE = d_nloss(self.y_out, y)

        # 2. Obliczanie gradientu dla warstwy wyjsciowej
        # 2.1 Gradient błędu w warstwie wyjściowej
        delta_out = dE * d_sigmoid(self.y_out)

        # 2.2 Gradient wag dla warstwy wyjściowej:
        grad_W_hidden_out = np.dot(delta_out.reshape(-1, 1), self.y_hidden.reshape(1, -1))

        # 3. Obliczenie gradientu dla warstwy ukrytej
        # 3.1 Błąd propagowany do warstwy ukrytej
        delta_hidden = np.dot(self.W_hidden_out.T, delta_out) * d_sigmoid(self.y_hidden)

        # 3.2 Gradient wag dla warstwy ukrytej
        grad_W_in_hidden = np.dot(delta_hidden.reshape(-1, 1), x.reshape(1, -1))

        # 4. Aktualizacja wag
        self.W_hidden_out -= self.LR * grad_W_hidden_out
        self.W_in_hidden -= self.LR * grad_W_in_hidden
        
    def train(self, x_set, y_set, iters): #forward -> strata (opc) -> backward -> aktualizacja wag (w backward jest)
        for i in range(iters):
            self.forward(x_set)
            if i % 100 == 0:
                E = np.sum(nloss(self.y_out, y))
                self.QUALITY_INDICATOR = E
                print(f'Wartość funkcji straty E, w iteracji {i}: {E}')
            self.backward(x_set, y_set)

#* Inicjalizacja instacji sieci neuronowej i wytrenowanie jej (poczatkowo 10000 iteracji)
nn = DlNet(x,y)

ITERATIONS = 15_000

losses = nn.train(x, y, ITERATIONS + 1)

yh = nn.predict(x) #WAZNE: tu umieszczono wyniki (y) z sieci

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
plt.plot(x,yh, 'b', linestyle='dotted')
plt.title("Liczba neuronów w warstwie ukrytej: " + str(nn.HIDDEN_L_SIZE) + ", Liczba iteracji: " + str(ITERATIONS) + ",\nWspół. uczenia: " + str(nn.LR) + ", Wskaźnik jakości aproks.: " + str("%.6f" % nn.QUALITY_INDICATOR))
plt.legend(['Funkcja aproksymowana', 'Perceptron dwuwarstwowy'], loc='upper left')
plt.show()
