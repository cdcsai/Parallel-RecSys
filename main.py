from RecommenderSystem import RecommenderSystem
from parameters import parameters


def main():
    recommender = RecommenderSystem(parameters)
    recommender.run(type="sgd")


if __name__ == '__main__':
    main()