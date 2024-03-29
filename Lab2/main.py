import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture(r"D:\2-bai tap cua dev\OpenCVProject\week2\videos\videoplayback.mp4")
roi = cv2.imread(r"D:\2-bai tap cua dev\OpenCVProject\week2\images\logoYoutube.png")
roiHSV = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# cv2.imshow("test", roi)
# r = cv2.selectROI("imageSelect", roi)
while True:
    ret, frame = cap.read()
    if ret:

        frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # roiHSV = cv2.cvtColor(roi[int(r[1]): int(r[1] + r[3]), int(r[0]`): int(r[0] + r[2])], cv2.COLOR_BGR2HSV)
        cv2.imshow("Roi image", roiHSV)


        #Use library
        # M = cv2.calcHist([roiHSV], [0], None, [180], [0, 180])
        # cv2.normalize(M, M, 0, 255, cv2.NORM_MINMAX)
        # B = cv2.calcBackProject([frameHSV], [0], M, [0, 180], 1)
        # cv2.normalize(B, B, 0, 255, cv2.NORM_MINMAX)

        # #Not use library
        I = cv2.calcHist([frameHSV], [0], None, [180], [0, 180])
        M = cv2.calcHist([roiHSV], [0], None, [180], [0, 180])
        R = M/(I + 1)
        h, s, v = cv2.split(frameHSV)
        B = R[h.ravel()]
        B = np.minimum(B, 1)
        B = B.reshape(frame.shape[0: 2])



        #Apply mask filter
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        imp_mask = cv2.filter2D(B, -1, kernel)
        imp_mask = np.uint8(imp_mask)
        imp_mask = cv2.normalize(imp_mask, None,0,255,cv2.NORM_MINMAX)
        cv2.imshow("Mask", imp_mask)
        _, thresh_mask = cv2.threshold(imp_mask, 210, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Mask", thresh_mask)
        mask = cv2.merge((thresh_mask, thresh_mask, thresh_mask))
        result = cv2.bitwise_or(frame, mask) #You can change this to and bitwise

        cv2.imshow("Video", result)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    else:
        break

# cv2.waitKey(10000)
cv2.destroyAllWindows()