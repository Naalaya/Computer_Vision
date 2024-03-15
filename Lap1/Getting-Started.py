import cv2

def add_text(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30)
    font_scale = 1
    color = (255, 255, 255)  # White color
    thickness = 2
    cv2.putText(frame, text, position, font, font_scale, color, thickness)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    add_text(frame, "Troll VietNam")
    cv2.imshow('Original Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
