from parameters import train_param, data_param
from TrainRecommender import TrainRecommender


class Experiment:
    def __init__(self, data_param, train_param):
        self.data_param = data_param
        self.train_param = train_param

    def stochastic_on(self):
        self.train_param["is_stochastic"] = True
        self.train_param["batch_size"] = None
        self.train_param["processes"] = None

    def batch_set(self, batch_size):
        self.train_param["is_stochastic"] = None
        self.train_param["batch_size"] = batch_size
        self.train_param["processes"] = None

    def processes_set(self, processes):
        self.train_param["is_stochastic"] = None
        self.train_param["batch_size"] = None
        self.train_param["processes"] = processes

    def convergence_processes(self, processes):
        run_times = []
        evals = []
        params = []

        for process in processes:

            self.processes_set(process)
            recsys = TrainRecommender(data_param, train_param)
            recsys.run()

            run_times.append(recsys.runtime)
            evals.append(recsys.test_perf)
            params.append([self.data_param, self.train_param])

        print("Done")