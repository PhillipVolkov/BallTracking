class SumQueue:
    '''
    Class that is a queue, which keeps track of its sum
    @params MAX_LEN max length of the queue
    '''
    def __init__(self, MAX_LEN):
        self.MAX_LEN = MAX_LEN
        self.queue = []
        self.sum = 0
    def add(self, elem):
        self.queue.insert(0, elem)
        self.sum += elem
        
        if (len(self.queue) >= self.MAX_LEN):
            last = self.queue.pop()
            self.sum -= last
    def getSum(self):
        return self.sum
    def getArray(self):
        return self.queue