import cv2
import numpy as np

cap = cv2.VideoCapture(0)
haar_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_alt.xml')
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)

        # Iterating through rectangles of detected faces
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()