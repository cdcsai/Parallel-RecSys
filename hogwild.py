import queue
import threading
import time
from multiprocessing.pool import Pool
from RecommenderSystem import RecommenderSystem
from multiprocessing.sharedctypes import Array
from ctypes import c_double


class HogWild(threading.Thread):

    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q


    def multiprocess(self, nb_workers, samples, x):


    def




def sgd_partial_update(x):
    pass




if __name__ == "__main__":
    threadList = ["Thread-" + str(i) for i in range(3)]
    nameList = list()
    for task in range(100):
        nameList.append(["{}".format(task)])

    workQueue = queue.Queue(100)

    threads = []
    threadID = 1

    # Create new threads
    for tName in threadList:
        thread = HogWild(threadID, tName, workQueue)
        thread.start()
        threads.append(thread)
        threadID += 1

    # Fill the queue
    for word in nameList:
        workQueue.put(word)

    # Wait for queue to empty
    while not workQueue.empty():
        pass

    # Notify threads it's time to exit
    exitFlag = 1

    # Wait for all threads to complete
    for t in threads:
        t.join()
    print("Exiting Main Thread")









