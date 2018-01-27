import numpy as np
import random
from tqdm import tqdm
from math import sqrt


class RecommenderSystem():
    def __init__(self, parameters):
        self.nb_users = parameters["nb_users"]
        self.nb_movies = parameters["nb_users"]
        self.R = self.initialize_r()
        self.df_size = parameters["df_size"]
        self.coord = np.empty((self.df_size, 2))
        self.test_ratio = parameters["test_ratio"]
        self.test_size = self.test_ratio * self.df_size
        self.landa = parameters["landa"]
        self.max_iter = parameters["max_iter"]

    def initialize_r(self):
        R = np.empty((self.nb_users, self.nb_movies))
        R[:] = np.NAN
        return R

    def load_file(self):
        coord_index = 0
        with open("ratings1M.dat") as file:
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
        test_set = np.zeros(self.test_size, 3)
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
        return test_set

    def learn(self):
        U = np.random.rand(self.k, self.nb_users)
        V = np.random.rand(self.nb_movies, self.k)
        U *= 0.1
        V *= 0.1
        for i in range(0, self.nb_users):
            U[0, i] = 1
        for i in range(0, self.nb_movies):
            V[i, 0] = 1
        I = np.identity(self.k)

        for _ in tqdm(range(self.max_iter)):
            # Estimation of U
            for i in range(0, self.nb_users):
                idx_IN_lines = np.nonzero(np.isnan(self.R[i, :]) == False)
                idx_IN_lines = np.asarray(idx_IN_lines)
                idx_IN_lines = idx_IN_lines.flatten()
                U[:, i] = np.linalg.solve(np.dot(np.transpose(V[idx_IN_lines, :]), V[idx_IN_lines, :]) + self.landa * I,
                                          np.dot(np.transpose(V[idx_IN_lines, :]), np.transpose(self.R[i, idx_IN_lines])))
            # Estimation of V
            for i in range(0, self.nb_movies):
                idx_IN_lines = np.nonzero(np.isnan(self.R[:, i]) == False)
                idx_IN_lines = np.asarray(idx_IN_lines)
                idx_IN_lines = idx_IN_lines.flatten()
                S = np.linalg.solve(np.dot(U[:, idx_IN_lines], np.transpose(U[:, idx_IN_lines])) + self.landa * I,
                                    np.dot(U[:, idx_IN_lines], self.R[idx_IN_lines, i]))
                V[i, :] = np.transpose(S)
        return U, V

    def evaluate(self, test_set, U, V):
        error = 0

        for s in range(0, self.test_size):
            i = int(test_set[s, 0])
            j = int(test_set[s, 1])
            RT = test_set[s, 2]
            Rij = np.dot(U[:, i], V[j, :])
            error += (Rij - RT) ** 2
        error /= self.test_size
        error = sqrt(error)
        return error

    def run(self):
        self.R = self.initialize_r()
        self.load_file()
        test_set = self.sample_data()
        U, V = self.learn()
        rmse = self.evaluate(test_set, U, V)
        print("The RMSE is" + str(rmse))