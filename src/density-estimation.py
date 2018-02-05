import sys
sys.path.append("/home/mehdi/PycharmProjects/Parallelizing-Word2Vec-/src/recsys/")

from recsys.RecommenderSystem import  RecommenderSystem
from parameters import data_param

recsys = RecommenderSystem(data_param)
recsys.load_file()
coord = recsys.coord

print()