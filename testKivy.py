import cv2
import numpy as np
from scipy import stats
import os
'''
https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
'''

dice_position = np.ones((0, 3), dtype='float32')
sensor_variance = 0
save_count = 0

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass

#def find_dice(contours, grey_image):
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
    #flags = cv2.KMEANS_RANDOM_centres    
    #compactness, labels, centres = cv2.kmeans(pixelpoints,
                                              #dice_count, None, criteria,
                                              #10, flags)
    
    #return centres.astype("int32")

def reduce_colours(image, K):
    """Not in use
    """
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

def k_means_dice(frame, setting):
    """ With a given number of dice in the image, attempt to find the dice centroids and convex hulls
    """
    
    if setting <= 0:
        setting = 1
    
    max_dice = 20
    dice_count = 6#min(setting, max_dice)

    blur = cv2.GaussianBlur(frame, (9,9), 0)
    
    threshold = 100
    # convert the image to grayscale
    grey_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    pixelpoints = np.zeros((0, 2))
    
    while pixelpoints.size == 0:
        ret, thresh = cv2.threshold(grey_image, threshold, 255, cv2.THRESH_BINARY_INV)       
    
        pixelpoints = np.transpose(np.nonzero(thresh)).astype(np.float32)
        threshold += 20
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 1.0)
    
    if dice_position.shape[0] >= dice_count and all(dice_position[:dice_count, 2] < 0.4):
        centres = dice_position[:dice_count, :2].astype(np.float32)
        
        knn = cv2.ml.KNearest_create()
        
        responses = np.arange(centres.shape[0])
        
        knn.train(centres, cv2.ml.ROW_SAMPLE, responses)
        ret, results, neighbours, dist = knn.findNearest(pixelpoints, 1)
        bestLabels = results.astype(np.int32)
        
        attempts = 1
        flags = cv2.KMEANS_USE_INITIAL_LABELS
        
    else:
        bestLabels = None
        centres = None
        attempts = 10
        flags = cv2.KMEANS_RANDOM_CENTERS
        
    compactness, labels, centres = cv2.kmeans(data=pixelpoints,
                                              K=dice_count,
                                              bestLabels=bestLabels,
                                              criteria=criteria,
                                              attempts=attempts,
                                              flags=flags,
                                              centers=centres)    

    a = labels[:100]
    b = np.unique(labels)
    c = labels.shape
    d = pixelpoints.shape
    
    pixelpoints = pixelpoints.astype('uint32')
    hull_dict = {i: None for i in range(centres.shape[0])}
    for i in range(centres.shape[0]):
        # The locations of the pixels in group i
        tmp = pixelpoints[np.nonzero(labels == i)[0], :]
        
        mask = np.zeros_like(thresh)
        # Set all pixels in group i to white
        mask[tmp[:, 0], tmp[:, 1]] = 255
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        hull_dict[i] = cv2.convexHull(np.concatenate(contours))
    
    return hull_dict, centres


def motion_model(frame):
    global old_frame, dice_position
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    dice_centres = dice_position[:, 0:2]
    dice_centres = np.flip(dice_centres, 1)
    
    # Update dice positions with motion model guess
    dice_centres_new, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, dice_centres, None, **lk_params)
    dice_position[:, 0:2] = np.flip(dice_centres_new, 1)
    dice_position[:, 2] *= 1.1
    
    old_frame = frame.copy()

def kalman_update(sensors):
    global dice_position
    
    if sensors.size == 0:
        dice_position[:, 2] *= 1.1
        return
    
    knn = cv2.ml.KNearest_create()
    
    # Run kNN algorithm to match sensor points to known points
    trainData = dice_position[:, 0:2]
    responses = np.arange(trainData.shape[0])
    
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    ret, results, neighbours, dist = knn.findNearest(sensors, 1)
    results = results.astype('uint8')
    
    initial_variance = 3
    sensor_var = 0.9
    variance_gain = 1.3
    
    # If no matching point from sensor, lower weight
    for point_i in set(responses).difference(set(results.flatten())):
        dice_position[point_i, 2] *= variance_gain
    
    for i, (x, y) in enumerate(sensors):
        max_distance = 100
        if dist[i] > max_distance:
            # Add new possible point
            new_point = np.array([x, y, initial_variance]).reshape((1, 3)).astype('float32')
            dice_position = np.append(dice_position, new_point, axis=0)
            
        else:
            response_i = results[i, 0]
            nearest_point = dice_position[response_i]
            K = nearest_point[2] / (sensor_var + nearest_point[2])
            dice_position[response_i, :2] = K * sensors[i] + (1 - K) * dice_position[response_i, :2]
            dice_position[response_i, 2] = sensor_var * K
        
    for i in range(dice_position.shape[0] - 1, -1, -1):
        max_var = 5
        if dice_position[i][2] > max_var:
            dice_position = np.delete(dice_position, i, axis=0)
            
    dice_position = dice_position[dice_position[:, 2].argsort()]
    

def process(frame, setting):
    global image_count, path, old_frame, dice_position, save_count
    #frame = cv2.resize(frame, (200, 200))
    
    image_processed = frame.copy() 
    
    hull_dict, centres = k_means_dice(frame, setting)
    
    if dice_position.shape[0] == 0:
        dice_position = np.concatenate((centres, np.ones((centres.shape[0], 1))), axis=1).astype('float32')
    
    motion_model(image_processed)
    
    kalman_update(centres)
    
    for centre in dice_position:
        cv2.circle(image_processed, (int(centre[1]), int(centre[0])), 5, (centre[2] * 25, 0, max(100, 255 - centre[2] * 100)), -1)
    
    image_processed = cv2.drawContours(image_processed, list(hull_dict.values()), -1, (0, 255, 0), 1)
    
    if save_count < 25:
        save_count += 1
        return image_processed
    
    print(len(hull_dict))
    for _, hull in hull_dict.items():
        
        
        mask = cv2.drawContours(np.zeros_like(frame), [hull], -1, 255, -1)
        mask_inv = cv2.bitwise_not(mask)
        
        #image_processed = cv2.bitwise_and(image_processed, image_processed, mask=mask_inv)
        #image_processed = cv2.add(image_processed, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)//2)
        
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])
        width = height = max(width, height)
    
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
        warped = cv2.warpPerspective(frame, M, (width, height))
        warped = cv2.resize(warped, (50, 50))
        
        cv2.imwrite(f'{path}\\{image_count}.jpg', warped)
        
        image_count += 1
    
    save_count = 0
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
_, old_frame = cap.read()

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