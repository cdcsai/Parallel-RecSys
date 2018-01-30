import queue
import threading
import time

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


def process_data(threadName, q, queueLock, workQueue):
    while not exitFlag:
        if not workQueue.empty():
            data = q.get()
            print("%s processing %s" % (threadName, data))
        else:

            time.sleep(1)


def sgd_partial_update(x, lbda):
    pass

def svm(x):
    pass


if __name__ == "__main__":

<<<<<<< HEAD
    threadList = ["Thread-" + str(i) for i in range(3)]
    nameList = list()
    for task in range(100):
        nameList.append(["{}".format(task)])

    queueLock = threading.Lock()
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
    queueLock.acquire()
    for word in nameList:
        workQueue.put(word)
    queueLock.release()

    # Wait for queue to empty
    while not workQueue.empty():
        pass

    # Notify threads it's time to exit
    exitFlag = 1

    # Wait for all threads to complete
    for t in threads:
        t.join()
    print("Exiting Main Thread")


=======
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
>>>>>>> 2f5da26d9ce8b14d0133c0070ccaaea6fd5d090d
