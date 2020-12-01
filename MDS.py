import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



def dis_matrix(data):
    dis = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            dis[i, j] = np.sum(np.power((data[i] - data[j]), 2))
    return dis

def s_matrix(dis):
    i_m = np.ones((dis.shape[0],1))
    first_term = dis @ i_m @ i_m.T / dis.shape[0]
    second_term = i_m @ i_m.T @ dis / dis.shape[0]
    third_term = i_m @ i_m.T @ dis @ i_m @ i_m.T / dis.shape[0]**2
    S = -0.5*(dis - first_term - second_term + third_term)
    return S

def MDS(arr):
    d_matrix = dis_matrix(arr)
    s = s_matrix(d_matrix)
    eig_val, eig_vec  = np.linalg.eig(s)
    # reverse sort
    idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]
    diag = np.diag(eig_val)
    result = eig_vec[:, :2].dot(diag[:2, :2])
    return result

if __name__ == '__main__':
    names = ['animal name','hair','feathers','eggs','milk','airborne',
             'aquatic','predator','toothed','backbone','breathes',
             'venomous','fins','legs','tail','domestic','catsize','type']
    data = pd.read_csv('./zoo.data', names=names)
    Name = data['animal name']
    Type = data['type']
    data = data.drop(columns=['animal name','type'])
    scaler = StandardScaler()
    dataset = scaler.fit_transform(data)
    result = MDS(dataset)
    plt.figure()
    plt.scatter(result[:, 0], result[:, 1], c=Type)
    plt.show()