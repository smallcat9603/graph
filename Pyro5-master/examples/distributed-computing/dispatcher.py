import queue
from Pyro5.api import expose, behavior, serve, register_dict_to_class
from workitem import Workitem


# For 'workitem.Workitem' we register a deserialization hook to be able to get these back from Pyro
register_dict_to_class("workitem.Workitem", Workitem.from_dict)


@expose
@behavior(instance_mode="single")
class DispatcherQueue(object):
    def __init__(self):
        self.workqueue = queue.Queue()
        self.resultqueue = queue.Queue()

    def putWork(self, item):
        self.workqueue.put(item)

    def getWork(self, timeout=5):
        try:
            return self.workqueue.get(block=True, timeout=timeout)
        except queue.Empty:
            raise ValueError("no items in queue")

    def putResult(self, item):
        self.resultqueue.put(item)

    def getResult(self, timeout=5):
        try:
            return self.resultqueue.get(block=True, timeout=timeout)
        except queue.Empty:
            raise ValueError("no result available")

    def workQueueSize(self):
        return self.workqueue.qsize()

    def resultQueueSize(self):
        return self.resultqueue.qsize()


# main program

serve({
    DispatcherQueue: "example.distributed.dispatcher"
})
