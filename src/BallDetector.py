from ultralytics import YOLO
from collections import deque
import supervision as sv
import numpy as np
import scipy.interpolate as intr
import matplotlib.pyplot as plt
import cv2
import time
import csv
import math
from FPS import FPS
from Plotter import Plotter
from SumQueue import SumQueue

class BallDetector:
    # set constants
    def __init__(self):
        self.BUFFER_SIZE = 64
        self.EXPORT_DATA = [["time", "x", "y", "x'", "y'", "x''", "y''", "y''/y'", "bounce?"]]
        self.YOLO_MODEL = 'src/models/v1.pt'
        self.PLOT = True
        
    def configure(self):
        # configure camera
        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
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
            self.xPlotLine, = ax[0,0].plot([], [], animated=True)
            self.xPlotNext, = ax[0,0].plot([], [], animated=True)
            self.xPlotBounce, = ax[0,0].plot([], [], animated=True, color="red", marker="o", markersize=5, linestyle="None")
            ax[0,0].set(title="X Plot", xlim=(-500,1000), ylim=(0,self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)))
            
            self.xFirstDerivativePlot, = ax[0,1].plot([], [], animated=True)
            self.xSecondDerivativePlot, = ax[0,1].plot([], [], animated=True)
            self.xSecondDerivativeAveragePlot, = ax[0,1].plot([], [], animated=True)
            ax[0,1].set(title="X' Plot", xlim=(-500,1000), ylim=(-.5,.5))
            
            self.yPlot, = ax[1,0].plot([], [], animated=True)
            self.yPlotLine, = ax[1,0].plot([], [], animated=True)
            self.yPlotNext, = ax[1,0].plot([], [], animated=True)
            self.yPlotBounce, = ax[1,0].plot([], [], animated=True, color="red", marker="o", markersize=5, linestyle="None")
            ax[1,0].set(title="Y Plot", xlim=(-500,1000), ylim=(0,self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            ax[1,0].invert_yaxis()
            
            self.yFirstDerivativePlot, = ax[1,1].plot([], [], animated=True)
            self.ySecondDerivativePlot, = ax[1,1].plot([], [], animated=True)
            self.ySecondDerivativeAveragePlot, = ax[1,1].plot([], [], animated=True)
            ax[1,1].set(title="Y' Plot", xlim=(-500,1000), ylim=(-.5,.5))
            ax[1,1].invert_yaxis()
            
            self.plotter = Plotter(fig.canvas, [self.xPlot, self.xPlotLine, self.xPlotNext, self.xFirstDerivativePlot, self.xSecondDerivativePlot, self.xSecondDerivativeAveragePlot, self.xPlotBounce, 
                                                self.yPlot, self.yPlotLine, self.yPlotNext, self.yFirstDerivativePlot, self.ySecondDerivativePlot, self.ySecondDerivativeAveragePlot, self.yPlotBounce])
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(.1)
        
    # get frame from cam
    def fetchFrame(self):
        ret, self.frame = self.cam.read()
        
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
        
            self.pts.appendleft([center, time.time()*1000.0-self.startTime, False])
            cv2.circle(self.frame, center, 10, (0, 0, 255), -1)
        else:
            self.pts.appendleft(None)

        if self.pts[0] is not None:
            self.EXPORT_DATA.append([self.pts[0][1], self.pts[0][0][0], self.pts[0][0][1], 0, 0, 0, 0, 0, False])
                
    # draws current trajectory
    def drawTrajectory(self, t, x, y):
        currTime = time.time()*1000.0-self.startTime
        for i in range(0, len(self.pts)):
            if self.pts[i] is not None:
                t.append(currTime-self.pts[i][1])
                x.append(self.pts[i][0][0])
                y.append(self.pts[i][0][1])
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
            
        return t, x, y
    
    # fits splines to current trajectory
    def fitTrajectory(self, t, x, y):
        if len(x) >= 5 and len(y) >= 5:
            # fit a parabola onto points
            xLine = intr.UnivariateSpline(t, x, k=3)#np.poly1d(np.polyfit(t,x,2))
            yLine = intr.UnivariateSpline(t, y, k=3)#np.poly1d(np.polyfit(t,y,2))
            
            xFirstDerivativeLine = xLine.derivative(n=1)
            xSecondDerivativeLine = xLine.derivative(n=2)
            yFirstDerivativeLine = xLine.derivative(n=1)
            ySecondDerivativeLine = yLine.derivative(n=2)
            
            # plot predicted line
            if (self.PLOT):
                tPlots = np.arange(0, 1000, 50)
                self.xPlotLine.set_data(tPlots, xLine(tPlots))
                self.yPlotLine.set_data(tPlots, yLine(tPlots))
                self.xFirstDerivativePlot.set_data(tPlots, xFirstDerivativeLine(tPlots))
                self.xSecondDerivativePlot.set_data(tPlots, xSecondDerivativeLine(tPlots))
                self.yFirstDerivativePlot.set_data(tPlots, yFirstDerivativeLine(tPlots))
                self.ySecondDerivativePlot.set_data(tPlots, ySecondDerivativeLine(tPlots))
                
            self.EXPORT_DATA[-1][3] = xFirstDerivativeLine(0)
            self.EXPORT_DATA[-1][4] = yFirstDerivativeLine(0)
            self.EXPORT_DATA[-1][5] = xSecondDerivativeLine(0)
            self.EXPORT_DATA[-1][6] = ySecondDerivativeLine(0)
            self.EXPORT_DATA[-1][7] = ySecondDerivativeLine(0)/yFirstDerivativeLine(0)
                
            return xFirstDerivativeLine(0), xSecondDerivativeLine(0), yFirstDerivativeLine(0), ySecondDerivativeLine(0)
        return 0, 0, 0, 0
        
    # bounce detection
    def detectBounce(self, xFirstDerivative, xSecondDerivative, yFirstDerivative, ySecondDerivative):
        averageX = self.averageAcceleration[0].getSum()/self.BUFFER_SIZE
        averageY = self.averageAcceleration[1].getSum()/self.BUFFER_SIZE
        
        if abs(xSecondDerivative) > (averageX+0.05)*2 or abs(ySecondDerivative) > (averageY+0.05)*2:
            self.pts[0][2] = True
            self.EXPORT_DATA[-1][-1] = True
            
        if (self.PLOT):
            currTime = time.time()*1000.0-self.startTime
            t = []
            x = []
            y = []
            for i in range(0, len(self.pts)):
                if self.pts[i] is not None and self.pts[i][2]:
                    t.append(currTime-self.pts[i][1])
                    x.append(self.pts[i][0][0])
                    y.append(self.pts[i][0][1])
            self.xPlotBounce.set_data(t, x)
            self.yPlotBounce.set_data(t, y)
            self.xSecondDerivativeAveragePlot.set_data([0, 1000], [averageX, averageX])
            self.ySecondDerivativeAveragePlot.set_data([0, 1000], [averageY, averageY])
        
        self.averageAcceleration[0].add(abs(xSecondDerivative))
        self.averageAcceleration[1].add(abs(ySecondDerivative))
    
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

            t, x, y = self.drawTrajectory(t, x, y)
            xFirstDerivative, xSecondDerivative, yFirstDerivative, ySecondDerivative = self.fitTrajectory(t, x, y)
            self.detectBounce(xFirstDerivative, xSecondDerivative, yFirstDerivative, ySecondDerivative)
            # self.predictNextPoint(t, x, y)
            
            if (self.PLOT):
                self.plotter.update()
            
    def writeBounceData(self):
        with open('src/data/bounce_data/bounce_data.csv', 'w', encoding='UTF8', newline='') as w:
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