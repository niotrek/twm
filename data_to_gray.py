import cv2
import os

img = cv2.imread("data_classificator/R/0.jpg")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)



# Wyświetlenie obrazu binarnego
cv2.imshow('Obraz binarny', binary_img)
cv2.waitKey(0)
cv2.destroyAllWindows()