#import cv2
#import numpy as np
#from scipy import stats
#'''
#https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
#'''
#def nothing(x):
    ## We need a callback for the createTrackbar function.
    ## It doesn't need to do anything, however.
    #pass

#def mask_dice():
    #for h,cnt in enumerate(contours):
        #mask = np.zeros(imgray.shape,np.uint8)
        #cv2.drawContours(mask,[cnt],0,255,-1)
        #mean = cv2.mean(im,mask = mask)    

#def find_dice(contours, grey_image):
    #eps = 0.1
    #contour_areas = np.zeros(len(contours))
    ##mask = np.full(len(contours), True)
    
    #for i, c in enumerate(contours):
        #contour_areas[i] = cv2.contourArea(c)
    
    #median_area = np.median(contour_areas)
    #total_area = sum(contour_areas)
    #max_dice = 20
    
    #dice_count = min(int(total_area / median_area), max_dice)
    
    #mask = np.zeros(grey_image.shape, np.uint8)
    #cv2.drawContours(mask, contours, -1, 255, -1)
    #pixelpoints = np.transpose(np.nonzero(mask)).astype("float32")
    #pixelpoints = np.flip(pixelpoints, 1)
    
    ##contour_points = np.concatenate(contours).reshape((-1, 2)).astype("float32")
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                #10, 1.0)
    #flags = cv2.KMEANS_RANDOM_CENTERS    
    #compactness, labels, centers = cv2.kmeans(pixelpoints,
                                              #dice_count, None, criteria,
                                              #10, flags)
    
    #return centers.astype("int32")

#def reduce_colours(image, K):
    #img_data = image / 255.0
    #img_data = img_data.reshape((-1, 3))

    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    #10, 1.0)
    #flags = cv2.KMEANS_RANDOM_CENTERS
    #img_data = img_data.astype(np.float32)
    #compactness, labels, centers = cv2.kmeans(img_data,
                                              #8, None, criteria,
                                              #10, flags)

    #new_colors = centers[labels].reshape((-1, 3))
    #image_processed = new_colors.reshape(frame.shape) * 255
    #image_processed = image_processed.astype('uint8')
    
    #return image_processed

#def process(frame):
    #image_processed = frame.copy()
    #blur = cv2.GaussianBlur(frame, (9,9), 0)

    ## convert the image to grayscale
    #grey_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    ## convert the grayscale image to binary image
    #ret, thresh = cv2.threshold(grey_image, 100, 255, 0)           

    ## find contours in the binary image
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
    
    #min_contour_area = 500
    #contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    #cv2.drawContours(image_processed, contours, -1, (0,255,0), thickness=-1)#, hierarchy=hierarchy, maxLevel=1)
    #if len(contours):
        #dice_centres = find_dice(contours, grey_image)
        
        #for centre in dice_centres:
            #cv2.circle(image_processed, tuple(centre), 5, (0, 0, 255), -1)
        
    #return image_processed

#cap = cv2.VideoCapture(0)

#cv2.namedWindow('Test')
##cv2.createTrackbar('Threshold1', 'Test', 0, 1200, nothing)
##cv2.createTrackbar('Threshold2', 'Test', 0, 1200, nothing)

##cv2.createTrackbar('Threshold', 'Test', 0, 255, nothing)
##cv2.createTrackbar('Canny Threshold 2', 'Hough Line Transform', 0, 1200, nothing)
##cv2.createTrackbar("Min Line Length", 'Hough Line Transform', 0, 100, nothing)
##cv2.createTrackbar("Max Line Gap", 'Hough Line Transform', 0, 100, nothing)

#while True:
    #_, img_original = cap.read()
    ##threshold1 = cv2.getTrackbarPos('Threshold1', 'Test')
    ##threshold2 = cv2.getTrackbarPos('Threshold2', 'Test')
    ###threshold_level = cv2.getTrackbarPos('Threshold', 'Test')
    
    ###img_original = cv2.imread("5 dice 1O.png")
    ##image_grey = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    ##image_processed = img_original.copy()
    
    ##contours, hierarchy = cv2.findContours(image_grey, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ##contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
    
    ##cv2.drawContours(image_processed, contours, -1, (0,255,0))#, hierarchy=hierarchy, maxLevel=1)
    ##if len(contours):
        ##dice_centres = find_dice(contours)
        
        ##for centre in dice_centres:
            ##cv2.circle(image_processed, tuple(centre), 5, (0, 0, 255), -1)
    
    #image_processed = process(img_original)
    ##cv2.imshow('Test', cv2.Canny(img_original,
                                 ##150,
                                 ##300))
    #cv2.imshow('Test', image_processed)

    #if cv2.waitKey(100) & 0xFF == ord('q'):
        #cv2.destroyAllWindows()
        #cap.release()
        #break