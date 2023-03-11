import numpy as np
import cv2
import time

cap = cv2.VideoCapture(
    'v4l2src device=/dev/video0 ! image/jpeg,framerate=61612/513,width=640,height=480 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true', cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    start_cp = time.time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    print(frame.shape)
    # Our operations on the frame come here
    # Display the resulting frame
    cv2.imshow('frame', frame)
    my_key = cv2.waitKey(1)
    if my_key == ord('q'):
        break
    elif my_key == ord('s'):
        cv2.imwrite('test.jpg', frame)
    print("Total time elapsed: {:.4f}".format(time.time() - start_cp))

# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()
