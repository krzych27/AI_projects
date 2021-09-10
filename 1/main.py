import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from scipy.spatial.distance import cdist
import skfuzzy as fuzz
from plots import plot
import math
import pandas as pd
import xlsxwriter

def _eta(u, d, m):
    u = u ** m
    n = np.sum(u * d, axis=1) / np.sum(u, axis=1)

    return n


def _update_clusters(x, u, m):
    um = u ** m
    v = um.dot(x.T) / np.atleast_2d(um.sum(axis=1)).T
    return v


def _hcm_criterion(x, v, n, m, metric):
    d = cdist(x.T, v, metric=metric)

    y = np.argmin(d, axis=1)

    u = np.zeros((v.shape[0], x.shape[1]))

    for i in range(x.shape[1]):
        u[y[i]][i] = 1

    return u, d


def _fcm_criterion(x, v, n, m, metric):
    d = cdist(x.T, v, metric=metric).T
    d = np.fmax(d, np.finfo(x.dtype).eps)

    exp = -2. / (m - 1)
    d2 = d ** exp

    u = d2 / np.sum(d2, axis=0, keepdims=1)

    return u, d


def _cmeans(x, c, m, e, max_iterations, criterion_function, metric="euclidean", v0=None, n=None):
    if not x.any() or len(x) < 1 or len(x[0]) < 1:
        print("Error: Data is in incorrect format")
        return

    S, N = x.shape

    if not c or c <= 0:
        print("Error: Number of clusters must be at least 1")

    if not m:
        print("Error: Fuzzifier must be greater than 1")
        return

    if v0 is None:
        xt = x.T
        v0 = xt[np.random.choice(xt.shape[0], c, replace=False), :]

    v = np.empty((max_iterations, c, S))
    v[0] = np.array(v0)
    u = np.zeros((max_iterations, c, N))
    t = 0

    while t < max_iterations - 1:

        u[t], d = criterion_function(x, v[t], n, m, metric)
        v[t + 1] = _update_clusters(x, u[t], m)

        if np.linalg.norm(v[t + 1] - v[t]) < e:
            break

        t += 1

    return v[t], v[0], u[t - 1], u[0], d, t


def hcm(x, c, e, max_iterations, metric="euclidean", v0=None):
    return _cmeans(x, c, 1, e, max_iterations, _hcm_criterion, metric, v0=v0)


def fcm(x, c, m, e, max_iterations, metric="euclidean", v0=None):
    return _cmeans(x, c, m, e, max_iterations, _fcm_criterion, metric, v0=v0)

def wskaznik(u):
    sum = 0
    for x in range(u.shape[0]):
        for y in range(u.shape[1]):
            sum = sum + u[x][y] ** 2

    sum = sum/(u.shape[1])
    return sum

figures = plot()

XA = figures.generate_figure_a()
XB = figures.generate_figure_b()
XC = figures.generate_figure_c()
XD = figures.generate_figure_d()
XE = figures.generate_figure_e()
XF = figures.generate_figure_f()


suma_1 = []
suma_2 = []
c = [2, 3, 4]

for i in c:
    v, v0, u, u0, d, t = hcm(XF.T, i, 0.1, 100)
    res = figures.generateResultPlot(XF, v, u, i, 'result_data_set_F_HCM_c'+str(i)+'.png','F','HCM',str(i))
    suma_1.append(res)
    suma_2.append(wskaznik(u))
    v, v0, u, u0, d, t = fcm(XF.T, i, 1.2, 0.1, 100)
    res = figures.generateResultPlot(XF, v, u, i, 'result_data_set_F_FCM_c'+str(i)+'.png','F','FCM',str(i))
    suma_1.append(res)
    suma_2.append(wskaznik(u))

df = pd.DataFrame(suma_1)
writer = pd.ExcelWriter('cryterium_1.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='cryterium_1', index=False)
writer.save()

df = pd.DataFrame(suma_2)
writer = pd.ExcelWriter('cryterium_2.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='cryterium_2', index=False)
writer.save()
