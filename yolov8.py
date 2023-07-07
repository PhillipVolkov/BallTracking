from ultralytics import YOLO
from collections import deque
import supervision as sv
import numpy as np
import cv2
import math
import time
import csv

BUFFER_SIZE = 16
EXPORT_DATA = [["time", "x", "y", "bounce?"]]

def main():
    # configure camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # specify model
    model = YOLO('best.pt')

    # specify annotator
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    # configure prev center point tracking
    pts = deque(maxlen=BUFFER_SIZE)

    # prediction loop
    nextPoint = (None, None)
    startTime = time.time()*1000.0
    prevFrameTime = startTime
    while True:
        #draw the bounding boxes and label
        ret, frame = cam.read()
        result = model(frame, conf=0.6)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        
        # filter out based on confidence
        balls = result.boxes

        #get the center point of the ball
        if len(balls) > 0:
            coords = balls[0].xywh.tolist()[0]
            center = ((int)(coords[0]), (int)(coords[1]))
        
            pts.appendleft([center, time.time()*1000.0-startTime])
            cv2.circle(frame, center, 10, (0, 0, 255), -1)
        else:
            pts.appendleft(None)

        if pts[0] is not None:
            EXPORT_DATA.append([pts[0][1], pts[0][0][0], pts[0][0][1], False])

        # trajectory tracking
        if len(balls) > 0:
            # loop over the set of tracked points
            t = []
            x = []
            y = []
            
            # bounce detection
            # if len(pts) >= 2 and pts[0] is not None and pts[1] is not None:
            #     prevDeviation = math.sqrt((pts[0][0][0]-pts[1][0][0])**2 + (pts[0][0][1]-pts[1][0][1])**2)
            #     deviation = 0
            #     if nextPoint[0] is not None and nextPoint[1] is not None:
            #         deviation = math.sqrt((nextPoint[0]-pts[0][0][0])**2 + (nextPoint[1]-pts[0][0][1])**2)
            #     percChange = 0
            #     if prevDeviation != 0:
            #         percChange = deviation/prevDeviation

            #     cv2.putText(frame, str(round(percChange, 2)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            #     if percChange >= 1:
            #         EXPORT_DATA[-1][3] = True
            #         cv2.putText(frame, "Bounce", (200,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            #         pts.clear()

            
            # draw current trajectory
            for i in range(0, len(pts)):
                if pts[i] is not None:
                    x.append(pts[i][0][0])
                    y.append(pts[i][0][1])
                    t.append(pts[i][1])
                # if either of the tracked points are None, ignore them
                if pts[i - 1] is None or pts[i] is None or i == 0:
                    continue
                # otherwise, compute the thickness of the line and draw the connecting lines
                thickness = int(np.sqrt(BUFFER_SIZE / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1][0], pts[i][0], (0, 0, 255), thickness)
                cv2.circle(frame, pts[i - 1][0], 5, (0, 0, 255), -1)
            
            #  predict the next points
            if len(x) >= BUFFER_SIZE/2 and len(y) >= BUFFER_SIZE/2:
                xLine = np.poly1d(np.polyfit(t,x,2))
                yLine = np.poly1d(np.polyfit(t,y,2))

                currTime = time.time()*1000.0-startTime
                nextPoint = (int(xLine(currTime)), int(yLine(currTime)))

                prevPoint = pts[0][0]
                for i in range(int(currTime), int(currTime)+500, 50):
                    point = (int(xLine(i)), int(yLine(i)))
                    cv2.circle(frame, point, 5, (0, 255, 255), -1)
                    cv2.line(frame, prevPoint, point, (0, 255, 255), 2)
                    prevPoint = point
            else:
                nextPoint = (None, None)

        frameTime = time.time()*1000.0-prevFrameTime
        cv2.putText(frame, str(int(1000.0/frameTime)), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        prevFrameTime = time.time()*1000.0

        cv2.imshow("yolov8", frame)

        key = cv2.waitKey(1)
        # space to record bounce
        if (key == 32):
            EXPORT_DATA[-1][3] = True

        # break with escape key
        if (key == 27):
            break
            
    cam.release()
    cv2.destroyAllWindows()

    with open('bounce_data/bounce_data.csv', 'w', encoding='UTF8', newline='') as w:
        writer = csv.writer(w)
        for row in EXPORT_DATA:
            writer.writerow(row)

if __name__ == "__main__":
    main()