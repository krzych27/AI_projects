import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import math
import pandas as pd


class plot:

    def generate_figure_a(self):
        plt.clf()
        plt.figure()
        X, y = ds.make_blobs(n_samples=50, random_state=0)
        transformation = [[0.6, -0.3], [-0.5, 0.2]]
        X_aniso = np.dot(X, transformation)

        X, y = ds.make_blobs(n_samples=50, random_state=0)
        transformation2 = [[-0.6, -0.3], [0.5, 0.2]]
        X_aniso2 = np.dot(X, transformation2)
        X_final = np.concatenate((X_aniso, X_aniso2))
        plt.scatter(X_final[:, 0], X_final[:, 1], marker='o', s=25)
        # plt.show()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend(title='Data set: a')
        plt.savefig('a.png')
        return X_final

    def generate_figure_b(self):
        plt.clf()
        plt.figure()
        X, Y = ds.make_moons(n_samples=100, shuffle=True, random_state=None)
        plt.scatter((-1) * X[:, 0], X[:, 1], marker='o', s=25)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend(title='Data set: b')
        plt.savefig('b.png')
        X[:,0] = (-1) * X[:,0]
        return X

    def generate_figure_c(self):
        plt.clf()
        X, Y = ds.make_circles(n_samples=100, factor=0.35)
        plt.scatter(X[:, 0], X[:, 1], marker='o', s=25)
        # plt.show()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend(title='Data set: c')
        plt.savefig('c.png')
        return X

    def generate_figure_d(self):
        plt.clf()
        plt.figure()

        X, y = ds.make_blobs(n_samples=10, random_state=0, centers=[(7, 1), (1, 1)], center_box=(0.0, 0.0),
                             cluster_std=1.5)
        transformation = [[-1, -0.6], [0.4, 0.2]]
        X_aniso = np.dot(X, transformation)
        X, y = ds.make_blobs(n_samples=10, random_state=0, centers=[(6, -4), (6, 6)], center_box=(0.0, 0.0),
                             cluster_std=1.5)
        transformation = [[1, -0.6], [-0.4, 0.2]]
        X_aniso2 = np.dot(X, transformation)
        X_aniso2 = X_aniso2.transpose()
        X_aniso2[1] = X_aniso2[1] + 1
        X_aniso2 = X_aniso2.transpose()
        X, y = ds.make_blobs(n_samples=10, random_state=0, centers=[(7, 1), (1, 1)], center_box=(0.0, 0.0),
                             cluster_std=1.5)
        transformation = [[-1, -0.6], [0.4, 0.2]]
        X_aniso3 = np.dot(X, transformation)
        X_aniso3 = X_aniso3.transpose()
        X_aniso3[0] = X_aniso3[0] + 20
        X_aniso3 = X_aniso3.transpose()
        X, Y = ds.make_blobs(n_samples=5, random_state=0, cluster_std=0.2, centers=1,
                             center_box=((-5.0, 0.0), (4.0, 5.0)))
        X_aniso4 = X
        X_aniso4 = X_aniso4.transpose()
        X_aniso4[1] = X_aniso4[1] - 8
        X_aniso4 = X_aniso4.transpose()
        X, Y = ds.make_blobs(n_samples=5, random_state=0, cluster_std=0.2, centers=1,
                             center_box=((-5.0, 0.0), (4.0, 10.0)))
        X_aniso5 = X
        X_aniso5 = X_aniso5.transpose()
        X_aniso5[1] = X_aniso5[1] - 8
        X_aniso5[0] = X_aniso5[0] + 11
        X_aniso5 = X_aniso5.transpose()
        X_final = np.concatenate((X_aniso, X_aniso2))
        X_final = np.concatenate((X_final, X_aniso3))
        X_final = np.concatenate((X_final, X_aniso4))
        X_final = np.concatenate((X_final, X_aniso5))
        plt.scatter(X_final[:, 0], X_final[:, 1], marker='o', s=25)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend(title='Data set: d')
        plt.savefig('d.png')
        return X_final

    def generate_figure_e(self):
        plt.clf()
        plt.figure()
        X, y = ds.make_blobs(n_samples=25, random_state=0, cluster_std=0.5, centers=1, center_box=(2.0, 2.0))
        X_aniso2 = X
        X, Y = ds.make_blobs(n_samples=25, random_state=0, cluster_std=0.1, centers=1,
                             center_box=((-5.0, 0.0), (4.0, 5.0)))
        X_aniso = X
        X_final = np.concatenate((X_aniso, X_aniso2))
        plt.scatter(X_final[:, 0], X_final[:, 1], marker='o', s=25)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend(title='Data set: e')
        plt.savefig('e.png')
        return X_final

    def generate_figure_f(self):
        plt.clf()
        plt.figure()
        X, y = ds.make_blobs(n_samples=25, random_state=0, center_box=(2.0, 2.0))
        transformation2 = [[-1, -0.6], [0.4, 0.2]]
        X_aniso2 = np.dot(X, transformation2)
        X, Y = ds.make_blobs(n_samples=25, random_state=0, cluster_std=0.2, centers=1, center_box=(6, -6))
        X_aniso = X
        X_final = np.concatenate((X_aniso, X_aniso2))
        plt.scatter(X_final[:, 0], X_final[:, 1], marker='o', s=25)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend(title='Data set: f')
        plt.savefig('f.png')
        return X_final

    def euclid_dist(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def miara_Xin_Zhao(self, X, m, c):
        x = X[:, 0]
        y = X[:, 1]

        res = 0

        sx = sum(x) / len(x)
        sy = sum(y) / len(y)

        for j in range(c):
            xx = x[m == j]
            yy = y[m == j]

            vx = sum(xx) / len(xx)
            vy = sum(yy) / len(yy)

            for i in range(len(xx)):
                a1 = self.euclid_dist(xx[i], yy[i], vx, vy)
                a2 = self.euclid_dist(xx[i], yy[i], sx, sy)
                res += a1 ** 2 - a2 ** 2

        return res

    def generateResultPlot(self, x, v, u, c, fileName, fig,type, number, labels=None):
        plt.clf()
        ax = plt.subplots()[1]
        cluster_membership = np.argmax(u, axis=0)

        for j in range(c):
            ax.scatter(
                x[0:, 0][cluster_membership == j],
                x[0:, 1][cluster_membership == j],
                alpha=0.5,
                edgecolors="none")

        ax.legend()
        ax.grid(True)
        plt.legend(title=('Result of data set ' + fig + ' for ' + type + ',c = ' + number))
        plt.savefig(fileName)
        res = self.miara_Xin_Zhao(x, cluster_membership, c)
        return res
