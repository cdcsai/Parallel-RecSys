import threading
import time

import queue
import threading
import time
import numpy as np

exitFlag = 0


class HogWild(threading.Thread):

    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q

    def run(self):
        print("Starting " + self.name)
        process_data(self.name, self.q)
        print("Exiting " + self.name)


def process_data(threadName, q):
    while not exitFlag:
        if not workQueue.empty():
            data = q.get()
            print("%s processing %s" % (threadName, data))
        else:

            time.sleep(1)


def sgd_partial_update(x, lbda):
   pass


if __name__ == "__main__":

   threadList = ["Thread-1", "Thread-2", "Thread-3"]
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