import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Abrindo banco de dados
dt = pd.read_csv("autos.csv",delimiter=",")

#Testando base de dados 
#print(dt['name'])

numEpocas = 10     # Vai subir / vai variar 

q = len(dt['index'])               # Número de padrões. Quantidade de carros cadastrados
print(f"Total de elementos: {q}")

eta = 0.01            # Taxa de aprendizado ( é interessante alterar para avaliar o comportamento)
m = 6                 # Número de neurônios na camada de entrada (year, gearbox, power, km, brand, repair)
N = 4                 # Número de neurônios na camada escondida.
L = 1                 # Número de neurônios na camada de saída.

# Carrega os dados de treinamento
year = pd.read_csv("autos.csv",delimiter=",",usecols=[8])
year = np.array(year)
year = np.transpose(year)
#print(year)teste tabela
gearbox = pd.read_csv("autos.csv",delimiter=",",usecols=[9])
gearbox = np.array(gearbox)
gearbox = np.transpose(gearbox)
#print(gearbox)teste tabela
power = pd.read_csv("autos.csv",delimiter=",",usecols=[10])
power = np.array(power)
power = np.transpose(power)
#print(power)teste tabela
km = pd.read_csv("autos.csv",delimiter=",",usecols=[12])
km = np.array(km)
km = np.transpose(km)
#print(km)teste tabela
brand =pd.read_csv("autos.csv",delimiter=",",usecols=[15])
brand = np.array(brand)
brand = np.transpose(brand)
#print(brand)teste tabela
repair = pd.read_csv("autos.csv",delimiter=",",usecols=[16])
repair = np.array(repair)
repair = np.transpose(repair)
#print(repair) teste tabela

#Valor desejado          
price = pd.read_csv("autos.csv",delimiter=",",usecols=[5])
price = np.array(price)
d = np.transpose(price)
#print(price)

W1 = np.random.random((N, m + 1)) #dimensões da Matriz de entrada
W2 = np.random.random((L, N + 1)) #dimensões da Matriz de saída

# Array para amazernar os erros.
E = np.zeros(q)
Etm = np.zeros(numEpocas) #Etm = Erro total médio ==> serve para acompanharmos a evolução do treinamento da rede

# bias
bias = 1

# Entrada do Perceptron.
X = np.vstack((year,gearbox, power, km, brand, repair))   # concatenação dos dois vetores

# ===============================================================
# TREINAMENTO.
# ===============================================================

for i in range(numEpocas): #repete o numero de vezes terminado, no caso 20
    for j in range(q): #repete o numero de "dados" existentes (nesse exemplo 13)
        
        # Insere o bias no vetor de entrada (apresentação do padrão da rede)
        Xb = np.hstack((bias, X[:,j])) #empilhamos pelo hstack junto ao bias e ficamos 
                                       #com unico vetor [bias peso PH]

        # Saída da Camada Escondida.
        o1 = np.tanh(W1.dot(Xb))            # Equações (1) e (2) juntas. 
                                            # (W1.dot(Xb))
                                            # np.tanh  ==> tangente hiperbólica
                                            # Geremos o vetor o1 = saida da camada intermediária

        # Incluindo o bias. Saída da camada escondida é a entrada da camada
        # de saída.
        o1b = np.insert(o1, 0, bias)

        # Neural network output
        Y = np.tanh(W2.dot(o1b))            # Equações (3) e (4) juntas.
                                            #Resulta na saída da rede neural
        
        e = d[j] - Y                        # Equação (5).

        # Erro Total.
        E[j] = (e.transpose().dot(e))/2     # Equação de erro quadrática.
        
        # Imprime o número da época e o Erro Total.
        # print('i = ' + str(i) + '   E = ' + str(E))
   
        # Error backpropagation.   
        # Cálculo do gradiente na camada de saída.
        delta2 = np.diag(e).dot((1 - Y*Y))          # Eq. (6)
        vdelta2 = (W2.transpose()).dot(delta2)      # Eq. (7)
        delta1 = np.diag(1 - o1b*o1b).dot(vdelta2)  # Eq. (8)

        # Atualização dos pesos.
        W1 = W1 + eta*(np.outer(delta1[1:], Xb))
        W2 = W2 + eta*(np.outer(delta2, o1b))
    
    #Calculo da média dos erros
    Etm[i] = E.mean()
   
plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.plot(Etm, color='b')
plt.plot(Etm)
plt.show()

# ===============================================================
# TESTE DA REDE.
# ===============================================================

Error_Test = np.zeros(q)

for i in range(q):
    # Insere o bias no vetor de entrada.
    Xb = np.hstack((bias, X[:,i]))

    # Saída da Camada Escondida.
    o1 = np.tanh(W1.dot(Xb))            # Equações (1) e (2) juntas.      
    #print(o1)
    
    # Incluindo o bias. Saída da camada escondida é a entrada da camada
    # de saída.
    o1b = np.insert(o1, 0, bias)

    # Neural network output
    Y = np.tanh(W2.dot(o1b))            # Equações (3) e (4) juntas.
    print(Y)
    
    Error_Test[i] = d[i] - (Y)
    
print(f'Erros: {Error_Test}')
print(np.round(Error_Test) - d) #aqui se ela acertou todas o vetor tem que estar zerado

