import pickle
from multiprocessing import cpu_count, Process, Queue
import matplotlib.patches as mptchs
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def man_dist_pbc(m, vector, shape=(10, 10)):
    dims = np.array(shape)
    delta = np.abs(m - vector)
    delta = np.where(delta > 0.5 * dims, np.abs(delta - dims), delta)
    return np.sum(delta, axis=len(m.shape) - 1)


class SOM(object):
    def __init__(self, x, y, alpha_start=0.6, seed=42):
        np.random.seed(seed)
        self.x = x
        self.y = y
        self.shape = (x, y)
        self.sigma = x / 2.
        self.alpha_start = alpha_start
        self.alphas = None
        self.sigmas = None
        self.epoch = 0
        self.interval = int()
        self.map = np.array([])
        self.indxmap = np.stack(np.unravel_index(np.arange(x * y, dtype=int).reshape(x, y), (x, y)), 2)
        self.distmap = np.zeros((self.x, self.y))
        self.winner_indices = np.array([])
        self.pca = None  # attribute to save potential PCA to for saving and later reloading
        self.inizialized = False
        self.error = 0.  # reconstruction error
        self.history = list()  # reconstruction error training history

    def initialize(self, data, how='pca'):
        self.map = np.random.normal(np.mean(data), np.std(data), size=(self.x, self.y, len(data[0])))
        if how == 'pca':
            eivalues = PCA(4).fit_transform(data.T).T
            for i in range(4):
                self.map[np.random.randint(0, self.x), np.random.randint(0, self.y)] = eivalues[i]

        self.inizialized = True

    def winner(self, vector):
        indx = np.argmin(np.sum((self.map - vector) ** 2, axis=2))
        return np.array([int(indx / self.x), indx % self.y])

    def cycle(self, vector):
        w = self.winner(vector)
        # get Manhattan distance (with PBC) of every neuron in the map to the winner
        dists = man_dist_pbc(self.indxmap, w, self.shape)
        # smooth the distances with the current sigma
        h = np.exp(-(dists / self.sigmas[self.epoch]) ** 2).reshape(self.x, self.y, 1)
        # update neuron weights
        self.map -= h * self.alphas[self.epoch] * (self.map - vector)

        print("Epoch %i;    Neuron [%i, %i];    \tSigma: %.4f;    alpha: %.4f" %
              (self.epoch, w[0], w[1], self.sigmas[self.epoch], self.alphas[self.epoch]))
        self.epoch = self.epoch + 1

    def fit(self, data, epochs=0, save_e=False, interval=1000, decay='hill'):
        self.interval = interval
        if not self.inizialized:
            self.initialize(data)
        if not epochs:
            epochs = len(data)
            indx = np.random.choice(np.arange(len(data)), epochs, replace=False)
        else:
            indx = np.random.choice(np.arange(len(data)), epochs)

        # get alpha and sigma decays for given number of epochs or for hill decay
        if decay == 'hill':
            epoch_list = np.linspace(0, 1, epochs)
            self.alphas = self.alpha_start / (1 + (epoch_list / 0.5) ** 4)
            self.sigmas = self.sigma / (1 + (epoch_list / 0.5) ** 4)
        else:
            self.alphas = np.linspace(self.alpha_start, 0.05, epochs)
            self.sigmas = np.linspace(self.sigma, 1, epochs)

        if save_e:  # save the error to history every "interval" epochs
            for i in range(epochs):
                self.cycle(data[indx[i]])
                if i % interval == 0:
                    self.history.append(self.som_error(data))
        else:
            for i in range(epochs):
                self.cycle(data[indx[i]])
        self.error = self.som_error(data)

    def transform(self, data):
        m = self.map.reshape((self.x * self.y, self.map.shape[-1]))
        dotprod = np.dot(np.exp(data), np.exp(m.T)) / np.sum(np.exp(m), axis=1)
        return (dotprod / (np.exp(np.max(dotprod)) + 1e-8)).reshape(data.shape[0], self.x, self.y)

    def distance_map(self, metric='euclidean'):
        dists = np.zeros((self.x, self.y))
        for x in range(self.x):
            for y in range(self.y):
                d = cdist(self.map[x, y].reshape((1, -1)), self.map.reshape((-1, self.map.shape[-1])), metric=metric)
                dists[x, y] = np.mean(d)
        self.distmap = dists / float(np.max(dists))

    def winner_map(self, data):
        wm = np.zeros(self.shape, dtype=int)
        for d in data:
            [x, y] = self.winner(d)
            wm[x, y] += 1
        return wm

    def _one_winner_neuron(self, data, q):
        q.put(np.array([self.winner(d) for d in data], dtype='int'))

    def winner_neurons(self, data):
        print("Calculating neuron indices for all data points...")
        queue = Queue()
        n = cpu_count() - 1
        for d in np.array_split(np.array(data), n):
            p = Process(target=self._one_winner_neuron, args=(d, queue,))
            p.start()
        rslt = []
        for _ in range(n):
            rslt.extend(queue.get(10))
        self.winner_indices = np.array(rslt, dtype='int').reshape((len(data), 2))

    def _one_error(self, data, q):
        errs = list()
        for d in data:
            w = self.winner(d)
            dist = self.map[w[0], w[1]] - d
            errs.append(np.sqrt(np.dot(dist, dist.T)))
        q.put(errs)

    def som_error(self, data):
        queue = Queue()
        for d in np.array_split(np.array(data), cpu_count()):
            p = Process(target=self._one_error, args=(d, queue,))
            p.start()
        rslt = []
        for _ in range(cpu_count()):
            rslt.extend(queue.get(50))
        return float(sum(rslt) / float(len(data)))

    def get_neighbors(self, datapoint, data, labels, d=0):
        if not len(self.winner_indices):
            self.winner_neurons(data)
        labels = np.array(labels)
        w = self.winner(datapoint)
        print("Winner neuron of given data point: [%i, %i]" % (w[0], w[1]))
        dists = np.array([man_dist_pbc(winner, w, self.shape) for winner in self.winner_indices]).flatten()
        return labels[np.where(dists <= d)[0]]

    def plot_point_map(self, data, targets, targetnames, filename=None, colors=None, markers=None, example_dict=None,
                       density=True, activities=None):
        if not markers:
            markers = ['o'] * len(targetnames)
        if not colors:
            colors = ['#EDB233', '#90C3EC', '#C02942', '#79BD9A', '#774F38', 'gray', 'black']
        if activities:
            heatmap = plt.get_cmap('coolwarm').reversed()
            colors = [heatmap(a / max(activities)) for a in activities]
        if density:
            fig, ax = self.plot_density_map(data, internal=True)
        else:
            fig, ax = plt.subplots(figsize=self.shape)

        for cnt, xx in enumerate(data):
            if activities:
                c = colors[cnt]
            else:
                c = colors[targets[cnt]]
            w = self.winner(xx)
            ax.plot(w[1] + .5 + 0.1 * np.random.randn(1), w[0] + .5 + 0.1 * np.random.randn(1),
                    markers[targets[cnt]], color=c, markersize=12)

        ax.set_aspect('equal')
        ax.set_xlim([0, self.x])
        ax.set_ylim([0, self.y])
        plt.xticks(np.arange(.5, self.x + .5), range(self.x))
        plt.yticks(np.arange(.5, self.y + .5), range(self.y))
        ax.grid(which='both')

        if not activities:
            patches = [mptchs.Patch(color=colors[i], label=targetnames[i]) for i in range(len(targetnames))]
            legend = plt.legend(handles=patches, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(targetnames),
                                mode="expand", borderaxespad=0.1)
            legend.get_frame().set_facecolor('#e5e5e5')

        if example_dict:
            for k, v in example_dict.items():
                w = self.winner(v)
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker='*', color='#FDBC1C', markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords='data', fontsize=18, fontweight='bold')

        if filename:
            plt.savefig(filename)
            plt.close()
            print("Point map plot done!")
        else:
            plt.show()

    def plot_density_map(self, data, colormap='Oranges', filename=None, example_dict=None, internal=False):
        wm = self.winner_map(data)
        fig, ax = plt.subplots(figsize=self.shape)
        plt.pcolormesh(wm, cmap=colormap, edgecolors=None)
        plt.colorbar()
        plt.xticks(np.arange(.5, self.x + .5), range(self.x))
        plt.yticks(np.arange(.5, self.y + .5), range(self.y))
        ax.set_aspect('equal')

        if example_dict:
            for k, v in example_dict.items():
                w = self.winner(v)
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker='*', color='#FDBC1C', markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords='data', fontsize=18, fontweight='bold')

        if not internal:
            if filename:
                plt.savefig(filename)
                plt.close()
                print("Density map plot done!")
            else:
                plt.show()
        else:
            return fig, ax

    def plot_class_density(self, data, targets, t=1, name='actives', colormap='Oranges', example_dict=None,
                           filename=None):
        targets = np.array(targets)
        t_data = data[np.where(targets == t)[0]]
        wm = self.winner_map(t_data)
        fig, ax = plt.subplots(figsize=self.shape)
        plt.pcolormesh(wm, cmap=colormap, edgecolors=None)
        plt.colorbar()
        plt.xticks(np.arange(.5, self.x + .5), range(self.x))
        plt.yticks(np.arange(.5, self.y + .5), range(self.y))
        plt.title(name, fontweight='bold', fontsize=28)
        ax.set_aspect('equal')
        plt.text(0.1, -1., "%i Datapoints" % len(t_data), fontsize=20, fontweight='bold')

        if example_dict:
            for k, v in example_dict.items():
                w = self.winner(v)
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker='*', color='#FDBC1C', markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords='data', fontsize=18, fontweight='bold')

        if filename:
            plt.savefig(filename)
            plt.close()
            print("Class density plot done!")
        else:
            plt.show()

    def plot_distance_map(self, colormap='Oranges', filename=None):
        if np.mean(self.distmap) == 0.:
            self.distance_map()
        fig, ax = plt.subplots(figsize=self.shape)
        plt.pcolormesh(self.distmap, cmap=colormap, edgecolors=None)
        plt.colorbar()
        plt.xticks(np.arange(.5, self.x + .5), range(self.x))
        plt.yticks(np.arange(.5, self.y + .5), range(self.y))
        plt.title("Distance Map", fontweight='bold', fontsize=20)
        ax.set_aspect('equal')
        if filename:
            plt.savefig(filename)
            plt.close()
            print("Distance map plot done!")
        else:
            plt.show()

    def plot_error_history(self, color='orange', filename=None):
        if not len(self.history):
            raise LookupError("No error history was found! Is the SOM already trained?")
        fig, ax = plt.subplots()
        ax.plot(range(0, self.epoch, self.interval), self.history, '-o', c=color)
        ax.set_title('SOM Error History', fontweight='bold')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Error', fontweight='bold')
        if filename:
            plt.savefig(filename)
            plt.close()
            print("Error history plot done!")
        else:
            plt.show()

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)