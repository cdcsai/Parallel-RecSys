from TrainRecommender import TrainRecommender
from parameters import data_param, train_param


def main():
    recommender = TrainRecommender(data_param, train_param)
    recommender.run()


if __name__ == '__main__':
    main()