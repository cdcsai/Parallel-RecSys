from RecommenderSystem import RecommenderSystem
from parameters import parameters


def main():
    recommender = RecommenderSystem(parameters)
    recommender.run()


if __name__ == '__main__':
    main()