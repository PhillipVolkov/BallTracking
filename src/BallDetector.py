from ultralytics import YOLO
from collections import deque
import supervision as sv
import numpy as np
import scipy.interpolate as intr
import matplotlib.pyplot as plt
import cv2
import time
import csv
from copy import copy
from FPS import FPS
from Plotter import Plotter
from SumQueue import SumQueue

class BallDetector:
    # set constants
    def __init__(self):
        self.BUFFER_SIZE = 64
        self.EXPORT_DATA = [["time", "x", "y", "x'", "y'", "bounce?"]]
        self.YOLO_MODEL = 'src/models/v1.pt'
        
        self.BOUNCE_DATA_SRC = 'src/data/bounce_data/bounce_data.csv'
        self.BOUNCE_FRAME_SRC = 'src/data/bounce_frames/'
        self.NUM_BOUNCE_PTS = 6
        self.BOUNCE_THRESHOLD = 1
        
        self.PLOT = True
        
    def configure(self):
        # configure camera
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.prevFrames = deque(maxlen=self.NUM_BOUNCE_PTS)
        self.numBounceFrames = 0
        
        # specify model
        self.model = YOLO(self.YOLO_MODEL)

        # specify annotator
        self.box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)

        # configure prev center point tracking
        self.pts = deque(maxlen=self.BUFFER_SIZE)
        self.nextPoint = (None, None)

        # configure fps tracker
        self.fps = FPS(0.25)
        # set the start time of the loop (ms)
        self.startTime = time.time()*1000.0
        
        # initialize bounce detection
        self.averageAcceleration = [SumQueue(self.BUFFER_SIZE), SumQueue(self.BUFFER_SIZE)]
        
        # configure plotter
        if (self.PLOT):
            fig, ax = plt.subplots(2, 2)
            fig.set_dpi(80)
            
            self.xPlot, = ax[0,0].plot([], [], animated=True)
            self.xPlotBounce, = ax[0,0].plot([], [], animated=True, color="red", marker="o", markersize=5, linestyle="None")
            ax[0,0].set(title="X Plot", xlim=(-500,1000), ylim=(0,self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)))
            
            self.xFirstDerivativePlot, = ax[0,1].plot([], [], animated=True)
            ax[0,1].set(title="X' Plot", xlim=(-500,1000), ylim=(-5,5))
            
            self.yPlot, = ax[1,0].plot([], [], animated=True)
            self.yPlotBounce, = ax[1,0].plot([], [], animated=True, color="red", marker="o", markersize=5, linestyle="None")
            ax[1,0].set(title="Y Plot", xlim=(-500,1000), ylim=(0,self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            ax[1,0].invert_yaxis()
            
            self.yFirstDerivativePlot, = ax[1,1].plot([], [], animated=True)
            ax[1,1].set(title="Y' Plot", xlim=(-500,1000), ylim=(-5,5))
            ax[1,1].invert_yaxis()
            
            self.plotter = Plotter(fig.canvas, [self.xPlot, self.xFirstDerivativePlot, self.xPlotBounce, 
                                                self.yPlot, self.yFirstDerivativePlot, self.yPlotBounce])
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(.1)
        
    # get frame from cam
    def fetchFrame(self):
        ret, self.frame = self.cam.read()
        self.prevFrames.appendleft(copy(self.frame))
        
    # run yolo classifier on frame
    def applyModel(self):
        #draw the bounding boxes and label
        result = self.model(self.frame, conf=0.3)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{self.model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]
        self.frame = self.box_annotator.annotate(scene=self.frame, detections=detections, labels=labels)
        
        # filter out based on confidence
        self.balls = result.boxes
    
    # get the center point of the ball
    def findCenterPoint(self):
        if len(self.balls) > 0:
            coords = self.balls[0].xywh.tolist()[0]
            center = ((int)(coords[0]), (int)(coords[1]))
        
            self.pts.appendleft([center, time.time()*1000.0-self.startTime, 0, 0, False])
            cv2.circle(self.frame, center, 10, (0, 0, 255), -1)
        else:
            self.pts.appendleft(None)

        if self.pts[0] is not None:
            self.EXPORT_DATA.append([self.pts[0][1], self.pts[0][0][0], self.pts[0][0][1], 0, 0, False])
                
    # draws current trajectory
    def drawTrajectory(self, t, x, y, dx, dy):
        currTime = time.time()*1000.0-self.startTime
        for i in range(0, len(self.pts)):
            if self.pts[i] is not None:
                t.append(currTime-self.pts[i][1])
                x.append(self.pts[i][0][0])
                y.append(self.pts[i][0][1])
                dx.append(self.pts[i][2])
                dy.append(self.pts[i][3])
            # if either of the tracked points are None, ignore them
            if self.pts[i - 1] is None or self.pts[i] is None or i == 0:
                continue
            # otherwise, compute the thickness of the line and draw the connecting lines
            thickness = int(np.sqrt(self.BUFFER_SIZE / float(i + 1)) * 2.5)
            cv2.line(self.frame, self.pts[i - 1][0], self.pts[i][0], (0, 0, 255), thickness)
            cv2.circle(self.frame, self.pts[i - 1][0], 5, (0, 0, 255), -1)
            
        if (self.PLOT):
            self.xPlot.set_data(t, x)
            self.yPlot.set_data(t, y)
            
        return t, x, y, dx, dy
    
    # calculates derivative for the current points
    def calcDerivative(self, t, x, y, prevDx, prevDy):
        if len(x) >= 2 and len(y) >= 2:
            dt = t[0]-t[1]
            dx = x[0]-x[1]
            dy = y[0]-y[1]
            
            prevDx[0] = dx/dt
            prevDy[0] = dy/dt
            self.pts[0][2] = prevDx[0]
            self.pts[0][3] = prevDy[0]
            
            # plot derivative graph
            if (self.PLOT):
                self.xFirstDerivativePlot.set_data(t, prevDx)
                self.yFirstDerivativePlot.set_data(t, prevDy)
                
            self.EXPORT_DATA[-1][3] = prevDx[0]
            self.EXPORT_DATA[-1][4] = prevDy[0]
                
            return prevDx, prevDy
        return [], [],
        
    # bounce detection
    def detectBounce(self, dx, dy):
        bounceDetected = False
        bounceFrame = 0
        
        if len(dx) < self.NUM_BOUNCE_PTS or len(dy) < self.NUM_BOUNCE_PTS or len(self.pts) < self.NUM_BOUNCE_PTS:
            return
        
        '''
        Invarient:
        1) First point has to be above the magnitutde threshold
        2) End point has to be above zero and first point has to be below zero
        3) First point needs to be greater in magnitutde then end point
        '''
        # only perform search if end point is currently above zero
        if dy[0] > 0:
            # search for criterion with first point up to 5 points away from current (end point)
            for i in range(1,self.NUM_BOUNCE_PTS):
                # if bounce already detected within range, break
                if self.pts[i] is not None and self.pts[i][-1]:
                    bounceDetected = False
                    break
                if abs(dy[i]) > self.BOUNCE_THRESHOLD and (dy[0] > 0 and dy[i] < 0) and (-dy[i] > dy[0]):
                    bounceDetected = True
                    break
                
        # Detect the exact bounce frame within range (min y value)
        if (bounceDetected):
            minY = self.pts[0][0][1]
            for i in range(1, self.NUM_BOUNCE_PTS):
                # (Y IS INVERTED)
                if self.pts[i] is not None and self.pts[i][0][1] > minY:
                    minY = self.pts[i][0][1]
                    bounceFrame = i
                    
            
        # Updates pts and export that bounce occured
        if (bounceDetected):
            self.pts[bounceFrame][-1] = True
            self.EXPORT_DATA[-(1+bounceFrame)][-1] = True
            cv2.imwrite(self.BOUNCE_FRAME_SRC+f'bounce_frame{self.numBounceFrames}.jpg', self.prevFrames[bounceFrame])
            self.numBounceFrames+=1
            
        # Plot the bounce
        if (self.PLOT):
            currTime = time.time()*1000.0-self.startTime
            t = []
            x = []
            y = []
            for i in range(0, len(self.pts)):
                if self.pts[i] is not None and self.pts[i][-1]:
                    t.append(currTime-self.pts[i][1])
                    x.append(self.pts[i][0][0])
                    y.append(self.pts[i][0][1])
            self.xPlotBounce.set_data(t, x)
            self.yPlotBounce.set_data(t, y)
    
    # predicts next point for the trajectory
    def predictNextPoint(self, t, x, y, xLine, yLine):
        if len(x) >= 5 and len(y) >= 5:
            # draw next points
            self.nextPoint = (int(xLine(0)), int(yLine(0)))
            prevPoint = self.pts[0][0]
            nextPoints = ([], [])
            for i in range(0, -500, -50):
                point = (int(xLine(i)), int(yLine(i)))
                nextPoints[0].append(point[0])
                nextPoints[1].append(point[1])
                cv2.circle(self.frame, point, 5, (0, 255, 255), -1)
                cv2.line(self.frame, prevPoint, point, (0, 255, 255), 2)
                prevPoint = point
            
            # PLOT next points
            if (self.PLOT):
                tPlots = range(0, -500, -50)
                self.xPlotNext.set_data(tPlots, nextPoints[0])
                self.yPlotNext.set_data(tPlots, nextPoints[1])
        else:
            self.nextPoint = (None, None)
            
    # trajectory tracking
    def calcTrajectory(self):
        if len(self.balls) > 0:
            # loop over the set of tracked points
            t = []
            x = []
            y = []
            dx = []
            dy = []

            t, x, y, dx, dy = self.drawTrajectory(t, x, y, dx, dy)
            xFirstDerivative, yFirstDerivative = self.calcDerivative(t, x, y, dx, dy)
            self.detectBounce(xFirstDerivative, yFirstDerivative)
            # self.predictNextPoint(t, x, y)
            
            if (self.PLOT):
                self.plotter.update()
            
    def writeBounceData(self):
        with open(self.BOUNCE_DATA_SRC, 'w', encoding='UTF8', newline='') as w:
            writer = csv.writer(w)
            for row in self.EXPORT_DATA:
                writer.writerow(row)
            
    def main(self):
        self.configure()
        
        # prediction loop
        while True:
            '''
            CALCULATIONS
            '''
            self.fetchFrame()
            self.applyModel()
            self.findCenterPoint()
            self.calcTrajectory()


            '''
            DISPLAY
            '''
            # display FPS
            cv2.putText(self.frame, str(self.fps.update()), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # display current frame
            cv2.imshow("yolov8", self.frame)
            
            
            '''
            ESCAPE LOOP
            '''
            key = cv2.waitKey(1)
            # break with escape key
            if (key == 27):
                break
                
        self.cam.release()
        cv2.destroyAllWindows()
        self.writeBounceData()

if __name__ == "__main__":
    ballDetector = BallDetector()
    ballDetector.main()