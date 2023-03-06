import cv2
import depthai
import numpy as np

pipeline = depthai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(60)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName('rgb')
cam_rgb.preview.link(xout_rgb.input)

# cam_depth = pipeline.createMonoDepth()
# cam_depth.setConfidenceThreshold(200)
# cam_depth.setOutputDepth(True)
# cam_depth.setPreviewSize(640, 480)

# xout_depth = pipeline.createXLinkOut()
# xout_depth.setStreamName('depth')
# cam_depth.out.link(xout_depth.input)

device = depthai.Device(pipeline)
device.startPipeline()

q_rgb = device.getOutputQueue(name='rgb')
# q_depth = device.getOutputQueue(name='depth')

while True:
    in_rgb = q_rgb.get()
#     in_depth = q_depth.get()

    frame = in_rgb.getCvFrame()
#     depth = in_depth.getFrame()

    # Define the range of white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    cnts, _ = cv2.findContours(mask_white.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)

        if cv2.contourArea(c) > 100:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.drawContours(frame, [c], 0, (0, 255, 0), 2)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            center = (cx, cy)

            cv2.circle(frame, center, 5, (255, 0, 0), -1)

            if cx < frame.shape[1] / 2 - 20:
                print("left")
            elif cx > frame.shape[1] / 2 + 20:
                print("right")
            else:
                print("forward")

            height, width, channels = frame.shape
            distance = height - cy
            if distance < height / 2 - 20:
                print("ascend")
            elif distance > height / 2 + 20:
                print("descend")
            else:
                print("hold")

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
