from TrainRecommender import TrainRecommender
from recsys.logs import write_to_json


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

    def convergence_processes(self, processes, same_obs=False):
        logs = []
        base_eval_it = self.train_param["eval_it"]
        base_max_iter = self.train_param["max_iter"]

        for process in processes:
            if same_obs :
                self.train_param["eval_it"] = int(base_eval_it / process)
                self.train_param["max_iter"] = int(base_max_iter / process)

            self.processes_set(process)
            print("processes %d", self.train_param["processes"])
            recsys = TrainRecommender(data_param=self.data_param, train_param=self.train_param)
            recsys.run()
            print(recsys)
            logs.append((self.train_param, recsys.get_dict()))
        write_to_json(logs)