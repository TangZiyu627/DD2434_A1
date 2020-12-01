import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.graph import graph_shortest_path

def distance_matrix(data):
    dis = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            dis[i, j] = np.sqrt(np.sum(np.power((data[i] - data[j]), 2)))
    return dis

def MDS(arr):
    eig_val, eig_vec = np.linalg.eig(arr)
    # reverse sort
    idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]
    diag = np.diag(eig_val)
    result = eig_vec[:, :2].dot(diag[:2, :2])
    return result

def k_matrix(dis, k):
    n_points = dis.shape[0]
    k_m = np.ones((n_points, n_points)) * np.inf
    for i in range(n_points):
        topk = np.argsort(dis[i, :])[:k + 1]
        k_m[i][topk] = dis[i][topk]
    return k_m


def iso(data, k):
    dis = distance_matrix(data)
    k_dis = k_matrix(dis, k)
    graph = graph_shortest_path(k_dis, directed=False)
    X = MDS(graph)
    return X


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
    result = iso(dataset,50)
    plt.figure()
    plt.scatter(result[:, 0], result[:, 1], c=Type)


    plt.show()
