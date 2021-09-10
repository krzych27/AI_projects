import pandas as pd
import matplotlib.pyplot as plt
import SimpSOM as simpSOM
import sklearn
import numpy as np
from som import SOM
from plots import plot
from sklearn.manifold import MDS


def feature_normalization(X, n):
    count_featrue = X.shape[-1]
    f_maxs = []
    f_mins = []

    for i in range(count_featrue):
        f_max = np.max(X[:, [i]])
        f_min = np.min(X[:, [i]])
        f_maxs.append(f_max)
        f_mins.append(f_min)

    for i in range(len(X)):
        x_normalizatoion = np.array([0.0] * n)
        for j in range(len(X[i])):
            f_max = f_maxs[j]
            f_min = f_mins[j]
            x_normalizatoion[j] = (X[i][j] - f_min) / (f_max - f_min)
        X[i] = x_normalizatoion
    return X


def make_XY(data):
    X, Y = [], []
    for d in data:
        d = d.split(',')
        X.append(d[1:])
        Y.append(d[0])
    #
    Y = [int(y) - 1 for y in Y]
    X = np.array([np.array([float(f) for f in x]) for x in X])
    return X, Y

def plot_clust(X, log,path):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    col = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'chocolate', 'indigo', 'silver', 'orange', 'gold', 'lime'];
    for idx, k in enumerate(log):
        for i in k:
            ax.scatter(X[i, 0], X[i, 1], X[i, 2], c=col[idx], marker='o')
    plt.savefig(path)
    plt.clf()


# XB = figures.generate_figure_b()
# XC = figures.generate_figure_c()
# XD = figures.generate_figure_d()
# XE = figures.generate_figure_e()
# XF = figures.generate_figure_f()

# data - wine(W)
# df_wine = pd.read_csv('data/wine.data')
# print('Wine\n')
# print(df_wine)
# data - glass(G)
# df_glass = pd.read_csv('data/glass.data')
# print('Glass\n')
# print(df_glass)
# # data - Wisconsin breast cancer
# df_wBc = pd.read_csv('data/breast-cancer-wisconsin.data')
# print('Wisconsin breast cancer\n')
# print(df_wBc)

#df_wine = df_wine.split('\n')[:-1]

if __name__ == "__main__":

    df_wine = pd.read_csv('data/wine.data')
    with open('data/wine.data','r',encoding='utf-8') as f:
         data = f.read()
    data = data.split('\n')[:-1]
    X, Y = make_XY(data)
    X = feature_normalization(X,13)

    figures = plot()
    X = figures.generate_figure_d()
    X = figures.convert_to_3d(X, 7)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
    plt.savefig('results/fig_d/figures_d.png')
    plt.clf()

    net = simpSOM.somNet(14, 14, X, PBC=True)
    net.train(0.01, 10000)
    net.save('results/fig_d/filename_weights')
    net.nodes_graph(colnum=0,path='results/fig_d/')
    net.diff_graph(path='results/fig_d/')

    labels = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'indigo']
    net.project(X, labels=labels,path='results/fig_d/')

    log = net.cluster(X, type='KMeans',path='results/fig_d/')
    plot_clust(X, log,'results/fig_d/figures_d_col.png')
    print('Print number of clusters')
    print(len(log))

    # MDS
    with open('data/wine.data', 'r', encoding='utf-8') as f:
        data = f.read()
    data = data.split('\n')[:-1]
    X, Y = make_XY(data)
    X = feature_normalization(X,13)
    res = MDS(2).fit(X).embedding_
    x = res[:, 0]
    y = res[:, 1]
    plt.scatter(x, y, color="green")
    plt.savefig('results/wine_simpSON/MDS.png')
    plt.clf()

    with open('data/wine.data','r',encoding='utf-8') as f:
        data = f.read()
    data = data.split('\n')[:-1]
    X, Y = make_XY(data)
    X = feature_normalization(X,13)

    CLASS_COUNT = 6
    FEATURE_COUNT = len(X[0])
    CLASS_COUNT = len(list(set(Y)))
    print('FEATURE_COUNT:%d' % FEATURE_COUNT)
    print('CLASS_COUNT:%d' % CLASS_COUNT)

    som = SOM(4, 4)
    som.fit(X, 10000, save_e=True, interval=100)
    som.plot_error_history(filename='results/som_error.png')

    som.plot_point_map(X, Y, ['Class %d' % (l + 1) for l in range(CLASS_COUNT)], filename='results/som.png')
    for i in range(CLASS_COUNT):
        som.plot_class_density(X, Y, t=i, name='Class %d' % (i + 1), filename='results/class_%d.png' % (i + 1))
    som.plot_distance_map(filename='results/distance_map.png')








