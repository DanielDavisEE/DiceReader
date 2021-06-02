import cv2
import numpy as np
import os, time
import matplotlib.pyplot as plt
'''
https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
'''

print(cv2.__version__)

dice_count = None
dice_position = np.ones((0, 3), dtype='float32')

def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass

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

def k_means_dice(pixelpoints):
    """ With a given number of dice in the image, attempt to find the dice centres using kmeans clustering
    """
    K = int(round(dice_count))
    if pixelpoints.size == 0 or K  == 0:
        return np.zeros((0, 2), dtype=np.float32)
    
    # If the appropriate number of dice positions exist with a low variance, use them as the start
    #     points for the kmeans algorithm. Use knn algorithm to determine best labels
    if dice_position.shape[0] >= K and all(dice_position[:K, 2] < 0.4):
        centres = dice_position[:K, :2].astype(np.float32)
        
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
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100, 1.0)
    
    compactness, labels, centres = cv2.kmeans(data=pixelpoints,
                                              K=K,
                                              bestLabels=bestLabels,
                                              criteria=criteria,
                                              attempts=attempts,
                                              flags=flags,
                                              centers=centres)
    
    assert centres.shape[1] == 2
    
    return centres


def find_dice_hulls(pixelpoints, frame_size, labels=None):
    """ Using the dice centres from the kalman posterior values and either a set of cluster labels or a
        thresholded image, find the convex hulls for each dice.
    """
    if dice_position.size == 0 or pixelpoints.size == 0 or int(round(dice_count)) == 0:
        return {}
    
    centres = dice_position[:int(round(dice_count)), :2].astype(np.float32)
    
    if labels is None:
        
        knn = cv2.ml.KNearest_create()
        
        responses = np.arange(centres.shape[0])
        
        knn.train(centres, cv2.ml.ROW_SAMPLE, responses)
        ret, results, neighbours, dist = knn.findNearest(pixelpoints, 1)
        labels = results.astype(np.int32)
    
    
    pixelpoints = pixelpoints.astype(np.int32)
    
    hull_dict = {i: None for i in range(centres.shape[0])}
    for i in range(centres.shape[0]):
        # The locations of the pixels in group i
        tmp = pixelpoints[np.nonzero(labels == i)[0], :]
        
        mask = np.zeros((frame_size[0], frame_size[1]), dtype=np.uint8)
        # Set all pixels in group i to white
        mask[tmp[:, 0], tmp[:, 1]] = 255
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            hull_dict[i] = cv2.convexHull(np.concatenate(contours))
        else:
            del hull_dict[i]
        
    return hull_dict


def motion_model(grey_image):
    """ Using optical flow as a motion model, update the estimated dice centres through dead reckoning.
    """
    global old_grey_image, dice_position
    
    if dice_position.size > 0:
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        dice_centres = dice_position[:, 0:2]
        dice_centres = np.flip(dice_centres, 1)
        
        # Update dice positions with motion model guess
        dice_centres_new, st, err = cv2.calcOpticalFlowPyrLK(old_grey_image, grey_image, dice_centres, None, **lk_params)
        dice_position[:, 0:2] = np.flip(dice_centres_new, 1)
        
    dice_position[:, 2] *= 1.1
    
    old_grey_image = grey_image.copy()


def kalman_update(sensors):
    """ Update the Kalman filters with the position estimations from the kmeans clustering.
    """
    
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
    
    if dice_position.shape[0] > 40:
        dice_position = dice_position[:min(round(dice_count * 2), 40), :]


def extract_dice_image(frame, hull, size=(50, 50)):
    rect = cv2.minAreaRect(hull)
    rect = (rect[0], (max(rect[1]), max(rect[1])), rect[2])
    
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
    warped = cv2.warpPerspective(frame, M, (width, height))
    return (int(rect[0][0]), int(rect[0][1])), box, cv2.resize(warped, (50, 50))
    

def find_blobs(im):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    
    ## Change thresholds
    #params.minThreshold = 10;
    #params.maxThreshold = 200;
    
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500
    params.maxArea = 50000
    
    ## Filter by Circularity
    #params.filterByCircularity = True
    #params.minCircularity = 0.1
    
    ## Filter by Convexity
    #params.filterByConvexity = True
    #params.minConvexity = 0.87
    
    ## Filter by Inertia
    #params.filterByInertia = True
    #params.minInertiaRatio = 0.01
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs.
    return detector.detect(im)


def find_dice_count(grey_image, thresh):
    global dice_count

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = sum([cv2.contourArea(cnt) for cnt in contours])
    
    adjustment_factor = 0.92 # Compensate for the tendency for the contours to give lower
                             #   areas than the circular blobs
    areas = [(kp.size/2)**2 * np.pi for kp in find_blobs(grey_image)]
    if areas:
        average_dice_size = np.average(areas) * adjustment_factor
    else:
        average_dice_size = total_area
        
    average_dice_size = min(average_dice_size, # Prevent dice number edge cases:
                            total_area)        #   Impossible to have between 0 and 1 dice
    
    new_dice_count = total_area / max(1, average_dice_size)
    if dice_count is None:
        dice_count = new_dice_count
    else:
        K = min(0.1 + 0.1 * abs(new_dice_count - dice_count), 0.5)
        dice_count = K * new_dice_count + (1 - K) * dice_count
    

def process(frame, setting):
    global image_count, path, old_frame, dice_position, record_training_images
    
    
    image_processed = frame.copy()
    
    # Preprocess image
    blur = cv2.GaussianBlur(frame, (9,9), 0)
    
    # convert the image to grayscale
    grey_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(grey_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((11, 11), dtype='uint8')
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
    find_dice_count(grey_image, thresh)
    
    new_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(new_image,contours,-1,(0, 255, 0),2)
    
    keypoints = find_blobs(grey_image)
    new_image = cv2.drawKeypoints(new_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #for kp in keypoints:
        #new_image = cv2.putText(new_image, str(int(kp.size)), tuple([int(x) for x in kp.pt]), cv2.FONT_HERSHEY_SIMPLEX, 
                                      #1, (0, 0, 255), thickness=2)
    #new_image = cv2.putText(new_image, str(dice_count), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                            #1, (0, 255, 0), thickness=2)
                            
    cv2.imwrite('Original.png', frame)
    cv2.imwrite('Processing.png', new_image)
    
    pixelpoints = np.transpose(np.nonzero(thresh)).astype(np.float32)
    centres = k_means_dice(pixelpoints)
    
    if dice_position.shape[0] == 0:
        dice_position = np.concatenate((centres, np.ones((centres.shape[0], 1))), axis=1).astype('float32')
    
    motion_model(grey_image)
    
    kalman_update(centres)
    
    # Add location dots
    #for y, x, var in dice_position:
        #red = (0, 0, 255 - var * 40)
        #cv2.circle(image_processed, (int(x), int(y)), 5, red, -1)
    
    
    hull_dict = find_dice_hulls(pixelpoints, frame.shape)
    
    for _, hull in hull_dict.items():
        
        hull_centre, hull_bbox, dice_image = extract_dice_image(image_processed, hull)
        
        # Draw bounding box around dice
        green = (0, 255, 0)
        cv2.drawContours(image_processed,[hull_bbox],0,green,2)
        
        if record_training_images:
            cv2.imwrite(f'{path}\\{image_count}.jpg', dice_image)
            image_count += 1
        else:
            # Predict outcomes
            blob = cv2.dnn.blobFromImage(cv2.cvtColor(dice_image, cv2.COLOR_BGR2GRAY), 1, (50, 50))
            
            net.setInput(blob)
            preds = net.forward()
            
            pred_classes = np.argsort(preds[0])[::-1] + 1
            
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # fontScale
            fontScale = 1            
            
            image_processed = cv2.putText(image_processed, str(pred_classes[0]), hull_centre, font, 
                                          fontScale, green, thickness=3)
            
        
    #image_processed = cv2.putText(image_processed, str(dice_count), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                  #1, (0, 255, 0), thickness=2)
    cv2.imwrite('Results.png', image_processed)
    return image_processed


record_training_images = False

if record_training_images:
    dice_type = 'd6'#f'image_folder_{count}'
    face = '6'
    path = f"{dice_type}\\{face}"
    
    try:
        os.mkdir(dice_type)
    except FileExistsError:
        pass
    
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    
    image_count = max([int(x.removesuffix('.jpg')) for x in os.listdir(path)]) + 1
else:
    net = cv2.dnn.readNetFromTensorflow(f'frozen_models\\frozen_graph.pb')

print('Opening Camera...')
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
print('Camera Open')
_, old_frame = cap.read()

# Preprocess image
blur = cv2.GaussianBlur(old_frame, (9,9), 0)
old_grey_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Test')

cv2.createTrackbar('Threshold', 'Test', 1, 6, nothing)

time_list = []
fps_list = []
count_list = []
filter_list = []

while True:
    _, img_original = cap.read()
    out.write(img_original)  # Write the frame to the output file.
    
    setting = cv2.getTrackbarPos('Threshold', 'Test')

    if setting <= 0:
        setting = 1    
    start_time = time.time()
    image_processed = process(img_original, setting)
    end_time = time.time()
    time_list.append(end_time - start_time)
    fps = 1 / np.average((time_list[-1 * min(len(time_list), 3):]))
    #fps_list.append(fps)
    #count_list.append(dice_count)
    #filter_list.append(dice_position.shape[0])
    
    image_processed = cv2.putText(image_processed, f"FPS: {fps:.2f}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 255, 0), thickness=2)
    cv2.imwrite('FPS.png', image_processed)
    cv2.imshow('Test', image_processed)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break

#plt.figure(1)
#plt.plot(filter_list)
#plt.plot(count_list)
#plt.plot(fps_list)
##plt.plot([6 for _ in range(len(filter_list))])
#plt.legend(('Number of Kalman filters', 'Estimate number of dice', 'FPS', 'Number of dice'))
#plt.show()

#plt.figure(2)    
#plt.plot(fps_list)
#plt.show()

