import cv2
import numpy as np

annotation = cv2.imread("cat_annotation.png")
annotation_t = np.float32(annotation)
annotation_t2 = annotation*255


print("annotation.shape",annotation.shape)

cv2.imshow("annotation",annotation)
cv2.imshow("annotation_t",annotation_t)
cv2.imshow("annotation_t2",annotation_t2)
cv2.imwrite("cat_annotation_2.png",annotation_t2)



annotation_1 = cv2.imread("cat_annotation_2.png")
annotation_2  = cv2.imread("person_annotation.png")

print("annotation_1.shape",annotation_1.shape)
print("annotation_2.shape",annotation_2.shape)

cv2.imshow("annotation_1",annotation_1)
cv2.imshow("annotation_2",annotation_2)
cv2.waitKey()

