import cv2
import numpy as np
# def add_roi(frame, roi):
#     h, w, _ = roi.shape
#     frame[50:h +50, 50:w+50 ] = roi
def add_text(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30)
    font_scale = 1
    color = (255, 255, 255)  # White color
    thickness = 2
    cv2.putText(frame, text, position, font, font_scale, color, thickness)


cap = cv2.VideoCapture(0)
# roi = cv2.imread(r"C:\Users\anpt2\Desktop\Code\Computer_Vision\Lap1\Image\Pandas.jpg")
# roi = cv2.resize(src=roi, dsize=(50, 50))
while True:
    ret, frame = cap.read()
    if not ret:
        break

    add_text(frame, "Troll VietNam")
    # add_roi(frame, roi)

    cv2.imshow('Original Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()