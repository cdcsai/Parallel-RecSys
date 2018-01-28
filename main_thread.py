if __name__ == "__main__":

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
