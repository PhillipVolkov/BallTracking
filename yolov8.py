from ultralytics import YOLO
from collections import deque
import supervision as sv
import numpy as np
import cv2
import math

BUFFER_SIZE = 16

def main():
    # configure camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

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
        
            pts.appendleft(center)
            cv2.circle(frame, center, 10, (0, 0, 255), -1)
        else:
            pts.appendleft(None)

        # trajectory tracking
        if len(balls) > 0:
            # loop over the set of tracked points
            t = []
            x = []
            y = []
            
            if pts[0] is not None and pts[1] is not None:
                prevDeviation = math.sqrt((pts[0][0]-pts[1][0])**2 + (pts[0][1]-pts[1][1])**2)
                if nextPoint[0] is not None and nextPoint[1] is not None and prevDeviation != 0:
                    deviation = math.sqrt((nextPoint[0]-pts[0][0])**2 + (nextPoint[1]-pts[0][1])**2)
                    percChange = deviation/prevDeviation
                    if percChange > 2:
                        cv2.putText(frame, "Bounce", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            for i in range(0, len(pts)):
                if pts[i] is not None:
                    x.append(pts[i][0])
                    y.append(pts[i][1])
                    t.append(i+1)
                # if either of the tracked points are None, ignore them
                if pts[i - 1] is None or pts[i] is None or i == 0:
                    continue
                # otherwise, compute the thickness of the line and draw the connecting lines
                thickness = int(np.sqrt(BUFFER_SIZE / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
                cv2.circle(frame, pts[i - 1], 5, (0, 0, 255), -1)
            
            #  predict the next points
            if len(x) >= BUFFER_SIZE/2 and len(y) >= BUFFER_SIZE/2:
                xLine = np.poly1d(np.polyfit(t,x,2))
                yLine = np.poly1d(np.polyfit(t,y,2))

                nextPoint = (int(xLine(0)), int(yLine(0)))

                prevPoint = pts[0]
                for i in range(0, -10, -1):
                    point = (int(xLine(i)), int(yLine(i)))
                    cv2.circle(frame, point, 5, (0, 255, 255), -1)
                    cv2.line(frame, prevPoint, point, (0, 255, 255), 2)
                    prevPoint = point
            else:
                nextPoint = (None, None)

        cv2.imshow("yolov8", frame)

        # break with escape key
        if (cv2.waitKey(30) == 27):
            break
            
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()