from recsys.RecommenderSystem import RecommenderSystem
from recsys.parallel import initialize_uv, parallel

from tqdm import tqdm
import numpy as np
import logging
import time

# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%a, %d %b %Y %H:%M:%S',
#                     filename='./src/log/recsys.log',
#                     filemode='w',
# )


class TrainRecommender(RecommenderSystem) :
    def __init__(self, data_param, train_param):
        super().__init__(data_param)
        self.regularization = train_param["regularization"]
        self.max_iter = train_param["max_iter"]
        self.k = train_param["k"]
        self.learning_rate = train_param["learning_rate"]

        self.eval_it = train_param["eval_it"]

        self.batch_size = train_param["batch_size"]
        self.is_stochastic = train_param["is_stochastic"]
        self.processes = train_param["processes"]

        self.test_perf = []
        self.runtime = 0

    def initialize(self):
        self.test_perf = []
        self.runtime = 0

    def check(self):
        nb_none = 0
        if self.processes is None :
            nb_none += 1
        if self.is_stochastic is None :
            nb_none += 1
        if self.batch_size is None :
            nb_none += 1

        try:
            assert nb_none == 2
        except AssertionError:
            print("Either processes, is_stochastic or batch_size should be not equal to None")
            raise

    def non_zero_idx(self, idx, is_u):
        if not is_u:
            return np.asarray(np.nonzero(np.isnan(self.R[idx, :]) == False)).flatten()
        else:
            return np.asarray(np.nonzero(np.isnan(self.R[:, idx]) == False)).flatten()

    def derivative(self, i, j, is_u, nonzero_len):
        # TODO : Check the derivative
        if is_u:
            return (np.dot(self.U[i, :], self.V[:, j]) - self.R[i, j]) * self.V[:, j] + \
                   self.regularization / nonzero_len * self.U[i, :]
        else:
            return (np.dot(self.U[j, :], self.V[:, i]) - self.R[j, i]) * self.U[j, :] + \
                   self.regularization / nonzero_len * self.V[:, i]

    def compute_gradients(self, is_u, index):
        non_zero_idx = self.non_zero_idx(index, is_u=is_u)
        nonzero_len = len(non_zero_idx)

        return [self.derivative(i, index, is_u=is_u, nonzero_len=nonzero_len)
                     for i in non_zero_idx]

    def gradient_step(self, is_u, d_index):
        non_zero_idx = self.non_zero_idx(d_index, is_u=is_u)
        nonzero_len = len(non_zero_idx)
        gradients = self.compute_gradients(is_u, d_index)

        if is_u :
            for s in range(nonzero_len):
                idx = non_zero_idx[s]
                self.U[idx, :] -= self.learning_rate * gradients[s]
        else:
            for s in range(nonzero_len):
                idx = non_zero_idx[s]
                self.V[:, idx] -= self.learning_rate * gradients[s]

    def learn_sgd(self, user_batch, movie_batch):
        """ Learn with Stochastic Graident Descent"""
        # Gradient descent over U
        idxs = np.random.choice(self.nb_users, user_batch, replace=False)
        for idx in idxs :
            self.gradient_step(is_u=False, d_index=idx)

        # Gradient descent over V
        idxs = np.random.choice(self.nb_movies, movie_batch, replace=False)
        for idx in idxs:
            self.gradient_step(is_u=True, d_index=idx)

    def learn_parallel(self):
        # Gradient descent over U
        idxs = np.random.choice(self.nb_users, self.processes, replace=False)
        args_list = [(False, idx) for idx in idxs]
        parallel(fun=self.gradient_step, args_list=args_list, n_processes=self.processes)

        # Gradient descent over V
        idxs = np.random.choice(self.nb_movies, self.processes, replace=False)
        args_list = [(True, idx) for idx in idxs]
        parallel(fun=self.gradient_step, args_list=args_list, n_processes=self.processes)

    def train(self):
        self.initialize()
        self.check()
        eval_time = 0

        start = time.time()
        self.U, self.V = initialize_uv(u_rows=self.nb_users, u_cols=self.k,
                                       v_cols=self.nb_movies)

        for it in tqdm(range(self.max_iter)):
            self.train_iter()

            if it % self.eval_it == 0:
                start_eval = time.time()
                self.test_perf.append(self.evaluate())
                end_eval = time.time()
                eval_time += end_eval - start_eval

        end = time.time()
        self.runtime = end - start - eval_time

    def train_iter(self):
        """ One iteration of training"""
        if self.processes is not None:
            logging.info('Training with SGD using %d', self.processes)
            self.learn_parallel()
        if self.is_stochastic is not None:
            logging.info('Training with SGD')
            self.learn_sgd(user_batch=self.nb_users, movie_batch=self.nb_movies)
        if self.batch_size is not None:
            logging.info('Training with mini-batch SGD, with a batch size of %d', self.batch_size)
            self.learn_sgd(user_batch=self.batch_size, movie_batch=self.batch_size)

    def get_dict(self):
        return {
            "eval_it": self.eval_it,
            "test_perf": self.test_perf,
            "runtime": self.runtime,
        }
