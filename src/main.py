from parameters import data_param, train_param
from recsys.Experiment import Experiment


def main():
    experiment = Experiment(data_param=data_param, train_param=train_param)
    experiment.convergence_processes(processes=[i for i in range(1, 2)])


if __name__ == '__main__':
    main()