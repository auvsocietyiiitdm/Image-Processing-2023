import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])

    # Threshold the frame to get only white colors
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Find contours in the mask
    cnts, _ = cv2.findContours(mask_white.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(cnts) > 0:
        # Find the largest contour
        c = max(cnts, key=cv2.contourArea)

        # Check if the contour is large enough to be valid
        if cv2.contourArea(c) > 100:
            # Compute the bounding box of the contour
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Draw the contour and the bounding box on the frame
            cv2.drawContours(frame, [c], 0, (0, 255, 0), 2)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

            # Compute the center of the bounding box
            M = cv2.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            center = (cx, cy)

            # Draw a circle at the center of the bounding box
            cv2.circle(frame, center, 5, (255, 0, 0), -1)

            # Send commands to align the ROV with the center of the bounding box
            if cx < frame.shape[1] / 2 - 20:
                print("left")
            elif cx > frame.shape[1] / 2 + 20:
                print("right")
            else:
                print("forward")

            # Find the height of the bounding box
            height, width, channels = frame.shape
            distance = height - cy
            if distance < height / 2 - 20:
                print("ascend")
            elif distance > height / 2 + 20:
                print("descend")
            else:
                print("hold")

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()