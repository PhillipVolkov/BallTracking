import time

class FPS:
    '''
    Class to measure fps of the camera stream
    @params POLLING_TIME time to update fps (in seconds)
    '''
    def __init__(self, POLLING_TIME):
        self.POLLING_TIME = POLLING_TIME
        self.startTime = 0
        self.frames = 0
        self.fps = 0
    def start(self):
        self.startTime = time.time()
    def update(self):
        self.frames += 1
        
        # if polling time has passed, update current fps
        if (time.time() - self.startTime >= self.POLLING_TIME):
            self.startTime = time.time()
            self.fps = self.frames / self.POLLING_TIME
            self.frames = 0
            return self.fps
        
        return self.fps