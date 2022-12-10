import numpy as np
import math
from random import randrange
import pickle

def derivative_function(x):
    return 1


def activation_function(x):
    if(x.all() > 0):
        return x
    else:
        return 0.000015 * pow(math.exp,x) - 1
    
def init_matrix_x(sequence,p,q,m):
    matrix = []
    for i in range (q - p):
        line = []
        for j in range(p):
            line.append(sequence[i+ j])
        for k in range(m):
            line.append(0)
        matrix.append(line)
    matrix = np.array(matrix)
    return matrix


def init_matrix_y(sequence,p,q):
    matrix = []
    for i in range (q - p):
        matrix.append(sequence[i+ p])
    matrix = np.array(matrix)
    return matrix

def init_w1(p,m):
    matrix = []
    for i in range(p+m):
        line = []
        for j in range(m):
            elem= randrange(10)/10-0.45343423
            line.append(elem)
        matrix.append(line)
    matrix = np.array(matrix)
    return matrix
    
def init_w2(m):
    matrix = []
    for i in range(m):
        line = []
        for j in range(1):
            elem= randrange(10)/10-0.45343423
            line.append(elem)
        matrix.append(line)
    matrix = np.array(matrix)
    return matrix

def initialise_all_matrix(sequence, p, m):
    q = len(sequence)
    x = []
    y = []
    x = init_matrix_x(sequence,p,q,m)
    y = init_matrix_y(sequence,p,q)
    x = x.reshape(x.shape[0], 1, x.shape[1])
    w1 = init_w1(p,m)
    w2 = init_w2(m)
    return x,y,w1,w2,q

def init_to_predict(sequence,p,m):
    q = len(sequence)
    x = init_matrix_x(sequence,p,q,m)
    x = x.reshape(x.shape[0], 1, x.shape[1])
    x[:, :, -m:] = 0
    y = init_matrix_y(sequence,p,q)
    k = y[-1].reshape(1)
    X = x[-1, 0, :-m]
    return k, X

def update_w1(w1,alpha,dy,x,w2,i):
    w1 -= alpha * dy * np.matmul(x[i].transpose(), w2.transpose()) * derivative_function(np.matmul(x[i], w1))
    return w1
    
def update_w2(w2,alpha,dy,hidden_layer):
    w2 -= alpha * dy * hidden_layer.transpose() * derivative_function(np.matmul(hidden_layer, w2))
    return w2
    
def step_results(hidden_layer,output,dy,error_all,w1,w2,x,y,i):
    hidden_layer = np.matmul(x[i], w1)
    output = np.matmul(hidden_layer, w2)
    dy = output - y[i]
    error_all += (dy ** 2)[0]
    return  hidden_layer,output,dy,error_all
    
def save_w1(w1,file_name):
    with open(file_name, "wb") as file:
        pickle.dump(w1, file)
        
def save_w2(w1,file_name):
    with open(file_name, "wb") as file:
        pickle.dump(w1, file)

def read_matrix_w1(file_name):
    with open(file_name, "rb") as file:
        matrix =  pickle.load( file)
        return matrix
    
def read_matrix_w2(file_name):
    with open(file_name, "rb") as file:
        matrix =  pickle.load( file)
        return matrix
    
    
def leraning(sequence: list, p: int, error: int, max_iter: int, m: int, alpha: float):
    error_all = 0
    k = 0
    x,y,w1,w2,q = initialise_all_matrix(sequence, p, m)
    # this code learn for each sample
    for j in range(max_iter):
        error_all = 0
        x[:, :, -m:] = 0
        for i in range(x.shape[0]):
            hidden_layer = activation_function(np.matmul(x[i], w1))
            output = activation_function(np.matmul(hidden_layer, w2))
            dy = output - y[i]
            w1 = update_w1(w1,alpha,dy,x,w2,i)
            w2 = update_w2(w2,alpha,dy,hidden_layer)
            try:
                x[i + 1][-m:] = hidden_layer
            except:
                pass
            # print("x=", x[i], "etalon", y[i], "result=", output)
        for i in range(x.shape[0]):
            hidden_layer,output,dy,error_all = step_results(hidden_layer,output,dy,error_all,w1,w2,x,y,i)
        
        print(j + 1, " ", error_all[0])
        if error_all <= error:
            break
    return w1,w2

def predict(w1,w2,sequence,p,m,predict_n):
    k, X  = init_to_predict(sequence,p,m)
    out = []
    for i in range(predict_n):
        X = X[1:]
        train = np.concatenate((X, k))
        X = np.concatenate((X, k))
        train = np.append(train, np.array([0] * m))
        hidden_layer = np.matmul(train, w1)
        output = np.matmul(hidden_layer, w2)
        k = output
        out.append(k[0])
    print("Следующее число",out)
    return out