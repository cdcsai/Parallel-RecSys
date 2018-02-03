import numpy as np
import random
from math import sqrt


class RecommenderSystem():
    def __init__(self, parameters):
        self.nb_users = parameters["nb_users"]
        self.nb_movies = parameters["nb_movies"]
        self.data_dir = parameters["data_dir"]

        self.df_size = parameters["df_size"]
        self.coord = np.empty((self.df_size, 2))
        self.test_ratio = parameters["test_ratio"]
        self.test_size = int(self.test_ratio * self.df_size)

        self.test_set = None
        self.coord = np.empty((self.df_size, 2))
        self.R = None
        self.U = None
        self.V = None

    def initialize_r(self):
        R = np.empty((self.nb_users, self.nb_movies))
        R[:] = np.NAN
        return R

    def load_file(self):
        coord_index = 0
        with open(self.data_dir) as file:
            for line in file:
                info = line.split("::", 3)
                user = int(info[0]) - 1
                item = int(info[1]) - 1
                rating = int(info[2])
                self.R[user, item] = rating
                self.coord[coord_index, 0] = user
                self.coord[coord_index, 1] = item
                coord_index += 1

    def sample_data(self):
        test_set = np.zeros((self.test_size, 3))
        sample = random.sample(range(0, self.df_size), self.test_size)
        sample_index = 0

        for w in range(0, self.test_size):
            i = int(self.coord[sample[w], 0])
            j = int(self.coord[sample[w], 1])
            test_set[w, 0] = i
            test_set[w, 1] = j
            test_set[w, 2] = self.R[i, j]
            self.R[i, j] = float('nan')
            sample_index += 1

        self.test_set = test_set

    def evaluate(self):
        error = 0

        for s in range(0, self.test_size):
            i = int(self.test_set[s, 0])
            j = int(self.test_set[s, 1])
            RT = self.test_set[s, 2]
            Rij = np.dot(self.U[i, :], self.V[:, j])
            error += (Rij - RT) ** 2
        error /= self.test_size
        error = sqrt(error)
        return error

    def run(self):
        self.R = self.initialize_r()
        self.load_file()
        self.sample_data()
        self.train()
        rmse = self.evaluate()
        print("The test RMSE is " + str(rmse))
        return rmse
