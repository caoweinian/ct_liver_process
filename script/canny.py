import os
import cv2
import time

if __name__ == '__main__':
    img = cv2.imread('/Volumes/LaCie_exFAT/dataset/medical_liver/paper_display/figure2/unique/162.png')
    t1 = time.time()
    img_canny = cv2.Canny(img, threshold1=256, threshold2=515)
    print(time.time() - t1)
    cv2.imshow('canny', img_canny)
    cv2.waitKey(0)
