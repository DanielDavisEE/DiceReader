import cv2
import numpy as np
from scipy import stats
import os
'''
https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
'''



def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass

def mask_dice():
    for h,cnt in enumerate(contours):
        mask = np.zeros(imgray.shape,np.uint8)
        cv2.drawContours(mask,[cnt],0,255,-1)
        mean = cv2.mean(im,mask = mask)    

def find_dice(contours, grey_image):
    contour_areas = np.zeros(len(contours))
    #mask = np.full(len(contours), True)
    
    for i, c in enumerate(contours):
        contour_areas[i] = cv2.contourArea(c)
    
    median_area = np.median(contour_areas)
    total_area = sum(contour_areas)
    max_dice = 20
    
    dice_count = min(int(total_area / median_area), max_dice)
    
    mask = np.zeros(grey_image.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)
    pixelpoints = np.transpose(np.nonzero(mask)).astype("float32")
    pixelpoints = np.flip(pixelpoints, 1)
    
    #contour_points = np.concatenate(contours).reshape((-1, 2)).astype("float32")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                10, 1.0)
    flags = cv2.KMEANS_RANDOM_centres    
    compactness, labels, centres = cv2.kmeans(pixelpoints,
                                              dice_count, None, criteria,
                                              10, flags)
    
    return centres.astype("int32")

def reduce_colours(image, K):
    if K <= 0:
        K = 1
    img_data = image / 255.0
    img_data = img_data.reshape((-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    10, 1.0)
    flags = cv2.KMEANS_RANDOM_centres
    img_data = img_data.astype(np.float32)
    compactness, labels, centres = cv2.kmeans(img_data,
                                              K, None, criteria,
                                              10, flags)

    new_colors = centres[labels].reshape((-1, 3))
    image_processed = new_colors.reshape(image.shape) * 255
    image_processed = image_processed.astype('uint8')
    
    return image_processed

def k_means_dice(thresh, setting):
    
    if setting <= 0:
        setting = 1
    
    max_dice = 20
    
    dice_count = min(6, max_dice)
    
    
    pixelpoints = np.transpose(np.nonzero(thresh)).astype("float32")
    #pixelpoints = np.flip(pixelpoints, 1)
    
    #contour_points = np.concatenate(contours).reshape((-1, 2)).astype("float32")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS    
    compactness, labels, centres = cv2.kmeans(pixelpoints,
                                              dice_count, None, criteria,
                                              10, flags)
    
    # Create false color image
    colors = np.random.randint(5, 250, size=(len(centres), 3), dtype=np.uint8)
    false_colors = colors[labels]
    a = centres.shape
    c = labels.shape
    pixelpoints = pixelpoints.astype('uint32')
    hull_dict = {i: None for i in range(centres.shape[0])}
    for i in np.unique(labels):
        # The locations of the pixels in group i
        tmp = pixelpoints[np.nonzero(labels == i)[0], :]
        
        mask = np.zeros_like(thresh)
        # Set all pixels in group i to white
        mask[tmp[:, 0], tmp[:, 1]] = 255
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        hull_dict[i] = cv2.convexHull(np.concatenate(contours))

    
    return hull_dict
    

def process(frame, setting):
    global image_count, path
    #frame = cv2.resize(frame, (200, 200))
    
    image_processed = frame.copy()
    blur = cv2.GaussianBlur(frame, (9,9), 0)

    # convert the image to grayscale
    grey_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey_image, 100, 255, cv2.THRESH_BINARY_INV)
    
    hull_dict = k_means_dice(thresh, setting)
    
    for _, hull in hull_dict.items():
        
        image_processed = cv2.drawContours(image_processed, [hull], -1, (0, 255, 0), 1)
        
        mask = cv2.drawContours(np.zeros_like(thresh), [hull], -1, 255, -1)
        mask_inv = cv2.bitwise_not(mask)
        
        #image_processed = cv2.bitwise_and(image_processed, image_processed, mask=mask_inv)
        #image_processed = cv2.add(image_processed, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)//2)
        
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])
    
        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
    
        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(blur, M, (width, height))
        
        cv2.imwrite(f'{path}\\{image_count}.jpg', warped)
        image_count += 1
    
    return image_processed
    
    ## convert the grayscale image to binary image
    #ret, thresh = cv2.threshold(grey_image, 100, 255, 0)

    ## find contours in the binary image
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
    
    #min_contour_area = 500
    #contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
    
    #cv2.drawContours(image_processed, contours, -1, (0,255,0), thickness=1)#, hierarchy=hierarchy, maxLevel=1)
    #if len(contours):
        #dice_centres = find_dice(contours, grey_image)
        
        #for centre in dice_centres:
            #cv2.circle(image_processed, tuple(centre), 5, (0, 0, 255), -1)
        
    #return image_processed

count = -1
for child in os.listdir():
    if os.path.isdir(child):
        count += 1

path = f'image_folder_{count}'
  
os.mkdir(path)
image_count = 0

cap = cv2.VideoCapture(0)

cv2.namedWindow('Test')

cv2.createTrackbar('Threshold', 'Test', 1, 10, nothing)

while True:
    _, img_original = cap.read()
    
    setting = cv2.getTrackbarPos('Threshold', 'Test')
    
    #img_original = cv2.imread("5 dice NO Broken.png")
    
    image_processed = process(img_original, setting)
    cv2.imshow('Test', image_processed)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        #cap.release()
        break