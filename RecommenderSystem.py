import numpy as np
import random
from tqdm import tqdm
from multiprocessing.pool import Pool
from multiprocessing.sharedctypes import Array
from ctypes import c_double
from math import sqrt


class RecommenderSystem():
    def __init__(self, parameters):
        self.nb_users = parameters["nb_users"]
        self.nb_movies = parameters["nb_movies"]
        self.nb_workers = parameters["nb_workers"]
        self.R = self.initialize_r()
        self.df_size = parameters["df_size"]
        self.coord = np.empty((self.df_size, 2))
        self.test_ratio = parameters["test_ratio"]
        self.test_size = int(self.test_ratio * self.df_size)

        self.regularization = parameters["regularization"]
        self.max_iter = parameters["max_iter"]
        self.k = parameters["k"]
        self.gradient_step = parameters["gradient_step"]

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
        return test_set

    def derivative(self, i, j, is_row, U, V, nonzero_len):
        assert self.R[i, j] is not np.nan
        if is_row:
            return (np.dot(U[i, :], V[:, j]) - self.R[i, j]) * V[:, j] + self.regularization / nonzero_len * U[i, :]
        else:
            return (np.dot(U[i, :], V[:, j]) - self.R[i, j]) * U[i, :] + self.regularization / nonzero_len * V[:, j]

    def non_zero_idx(self, idx, is_row):
        if is_row:
            return np.asarray(np.nonzero(np.isnan(self.R[idx, :])==False)).flatten()
        else:
            return np.asarray(np.nonzero(np.isnan(self.R[:, idx])==False)).flatten()

    def sgd_U_V(self, idxs):
        # V = np.random.rand(self.k, self.nb_movies) * .1
        V = Array(c_double, np.random.rand(self.k, self.nb_movies) * .1, lock=False)
        # U = np.random.rand(self.nb_users, self.k) * .1
        U = Array(c_double, np.random.rand(self.nb_users, self.k) * .1, lock=False)
        for i in range(0, self.nb_movies):
            V[0, i] = 1
        for i in range(0, self.nb_users):
            U[i, :] = 1
            # Gradient descent over V
        for _ in tqdm(range(self.max_iter)):

            idxs = np.random.choice(self.nb_movies, 4, replace=False)
            for j in idxs:
                non_zero_idx = self.non_zero_idx(j, is_row=False)
                gradients = [self.derivative(i, j, is_row=True, U=U, V=V, nonzero_len=len(non_zero_idx))
                             for i in non_zero_idx]
                for s in range(len(non_zero_idx)):
                    idx = non_zero_idx[s]
                    U[idx, :] -= self.gradient_step * gradients[s]



        for _ in tqdm(range(self.max_iter)):
            # Gradient descent over U
            idxs = np.random.choice(self.nb_users, 4, replace=False)
            for i in idxs:
                non_zero_idx = self.non_zero_idx(i, is_row=True)
                gradients = [self.derivative(i, j, is_row=False, U=U, V=V, nonzero_len=len(non_zero_idx))
                             for j in non_zero_idx]
                for s in range(len(non_zero_idx)):
                    idx = non_zero_idx[s]
                    U[:, idx] -= self.gradient_step * gradients[s]
        return V, U

    def learn_sgd_hogwild(self):
        p = Pool(self.nb_workers)
        results = p.apply_async(self.sgd_U_V, idxs)

        # batch_size = 1
        # examples = [None] * int(X.shape[0] / float(batch_size))
        # for k in range(int(X.shape[0] / float(batch_size))):
        #     Xx = X[k * batch_size: (k + 1) * batch_size, :].reshape((batch_size, X.shape[1]))
        #     yy = y[k * batch_size: (k + 1) * batch_size].reshape((batch_size, 1))
        #     examples[k] = (Xx, yy)




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
            Rij = np.dot(U[i, :], V[:, j])
            error += (Rij - RT) ** 2
        error /= self.test_size
        error = sqrt(error)
        return error

    def run(self, type):
        self.R = self.initialize_r()
        self.load_file()
        test_set = self.sample_data()
        if type == "efficient":
            U, V = self.learn()
        if type == "sgd":
            U, V = self.learn_sgd()
        rmse = self.evaluate(test_set, U, V)
        print("The RMSE is " + str(rmse))