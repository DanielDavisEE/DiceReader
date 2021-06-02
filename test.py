import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('Picture1.jpg', cv2.IMREAD_GRAYSCALE)

## global thresholding
#ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
## Otsu's thresholding
#ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
## Otsu's thresholding after Gaussian filtering
#blur = cv2.GaussianBlur(img,(9,9),0)
#ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
## plot all the images and their histograms
#images = [img, 0, th1,
          #img, 0, th2,
          #blur, 0, th3]
#titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          #'Original Noisy Image','Histogram',"Otsu's Thresholding",
          #'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

#plt.figure(0)
#for i in range(3):
    #plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    #plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    #plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    #plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    #plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    #plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
#plt.show()


# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
#blur = cv2.GaussianBlur(img,(9,9),0)
#ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((11, 11), dtype='uint8')
th3 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

# plot all the images and their histograms
images = [img, th1, th2]
titles = ['Original Image','Histogram','Global Thresholding (v=127)', "Otsu's Thresholding"]

plt.figure(0)
plt.subplot(2, 2, 1),plt.imshow(images[0],'gray')
plt.title(titles[0]), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2),plt.hist(images[0].ravel(),256)
plt.title(titles[1]), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3),plt.imshow(images[1],'gray')
plt.title(titles[2]), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4),plt.imshow(images[2],'gray')
plt.title(titles[3]), plt.xticks([]), plt.yticks([])

plt.show()

cv2.imwrite('Otsu.png', th2)

# Compare adaptiveThreshold methods
#img = cv2.medianBlur(img,5)
#ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            #cv2.THRESH_BINARY,11,2)
#th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY,11,2)
#titles = ['Original Image', 'Global Thresholding (v = 127)',
            #'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
#images = [img, th1, th2, th3]

#plt.figure(1)
#for i in range(4):
    #plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    #plt.title(titles[i])
    #plt.xticks([]),plt.yticks([])
#plt.show()