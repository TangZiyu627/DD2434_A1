import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
    pca = PCA(n_components=3)
    pca.fit(dataset)
    pca_data = pca.transform(dataset)
    plt.figure()
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=Type)
    plt.show()





