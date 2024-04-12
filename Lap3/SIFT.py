import cv2
import matplotlib.pyplot as plt
import numpy as np

sourceImage = cv2.imread(r"/Users/a410/Desktop/Computer_Vision/Lap3/e6edb023caa43d9a490931ea9b61160b.jpg")
targetImage = cv2.imread(r"/Users/a410/Desktop/Computer_Vision/Lap3/e6edb023caa43d9a490931ea9b61160b.png")

sift = cv2.SIFT_create()

sourceImageGray = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2GRAY)
targetImage = cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(sourceImageGray, None)
kp2, des2 = sift.detectAndCompute(targetImage, None)

index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good_matches = [[0, 0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < 0.6 * n.distance:
        good_matches[i] = [1, 0]

result = cv2.drawMatchesKnn(sourceImage, kp1, targetImage, kp2, matches, None,  matchColor=(0, 155, 0),
                             singlePointColor=(255, 0, 0),
                             matchesMask=good_matches,
                             flags=0)
cv2.imshow("SIFT Matching",result)
cv2.waitKey(5000000)
cv2.destroyAllWindows()