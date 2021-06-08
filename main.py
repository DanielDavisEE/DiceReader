import cv2
import numpy as np

import kivy

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
from kivy.properties import *

# Set the following variables if the aim is to collect training images for a learning algorithm.
# Only a single dice should be in frame at a time and only focussed on one face value.

RECORD_TRAINING_IMAGES = False # Bool value which chooses betweening recording training images and classification
DICE_TYPE = 'd6'               # The type of dice, only used for recording images
DICE_FACE = '1'                # The face of the dice when recording images

class CameraWidget(Widget):
    
    img_frame = ObjectProperty(force_dispatch=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Init class info
        self.cap = cv2.VideoCapture('./10s_clip.avi')
        self.dice_count = None
        self.dice_position = np.ones((0, 3), dtype='float32')
        
        # Read one image frame for setup purposes
        ret, frame = self.cap.read()
        
        if not ret:
            raise Exception("No frame returned by VideoCapture, capture device likely incorrect.")
        
        # Preprocess image
        blur = cv2.GaussianBlur(frame, (9,9), 0)
        self.old_grey_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)        
        
        if RECORD_TRAINING_IMAGES:
            # Setup info for training image collection
            dice_type = DICE_TYPE
            face = DICE_FACE
            self.path = f"{dice_type}\\{face}"
            
            try:
                os.mkdir(dice_type)
            except FileExistsError:
                pass
            
            try:
                os.mkdir(self.path)
            except FileExistsError:
                pass
            
            self.image_count = max([int(x.removesuffix('.jpg')) for x in os.listdir(self.path)]) + 1
        else:
            # Setup up pretrained TensorFlow classification net in OpenCV
            self.net = cv2.dnn.readNetFromTensorflow(f'frozen_models\\frozen_graph.pb')
        
        # Init Kivy info
        self.screen_width = frame.shape[1]
        self.screen_height = frame.shape[0]
        self.texture_array = Texture.create(size=(self.screen_width, self.screen_height))
        Window.size = (self.screen_width, self.screen_height)
    
    
    def capture_frame(self, *_):
        """ This method is scheduled by the widget to run at a fixed rate and updates the captured images
        """
        
        ret, frame = self.cap.read()  # Read an image from the frame.
        if ret:
            
            frame = self.process(frame) # Runs the computer vision algorithms
            
            # Convert image to format for widget to display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.img_frame = cv2.flip(frame, 0)
                
            
    def on_img_frame(self, *_):
        """ This method is triggered by self.img_frame changing and displays it on the widget
        """
        
        arr = self.img_frame.reshape((-1))
        arr = arr.astype('ubyte')
        
        # Blit the array onto a Kivy texture
        self.texture_array.blit_buffer(arr, colorfmt='rgb', bufferfmt='ubyte')      
        
        # Print the texture to the widget canvas
        with self.canvas:
            Rectangle(texture=self.texture_array, pos=(0, 0), 
                      size=(self.screen_width, self.screen_height))
            
            
    def close_camera(self):
        """ Release the capture device on application close
        """
        self.cap.release()


    def k_means_dice(self, pixelpoints):
        """ With a given number of dice in the image, attempt to find the dice centres using kmeans clustering
        """
        K = int(round(self.dice_count))
        if pixelpoints.size == 0 or K  == 0:
            return np.zeros((0, 2), dtype=np.float32)
        
        # If the appropriate number of dice positions exist with a low variance, use them as the start
        #     points for the kmeans algorithm. Use knn algorithm to determine best labels
        if self.dice_position.shape[0] >= K and all(self.dice_position[:K, 2] < 0.4):
            centres = self.dice_position[:K, :2].astype(np.float32)
            
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
        
        # Run k-means algorithm on coordinates of 'dice' pixels
        compactness, labels, centres = cv2.kmeans(data=pixelpoints,
                                                  K=K,
                                                  bestLabels=bestLabels,
                                                  criteria=criteria,
                                                  attempts=attempts,
                                                  flags=flags,
                                                  centers=centres)
        
        assert centres.shape[1] == 2 # Testing that at least one 'dice' pixel was input
        
        return centres
    
    
    def find_dice_hulls(self, pixelpoints, frame_size, labels=None):
        """ Using the dice centres from the kalman posterior values and either a set of cluster labels or a
            thresholded image, find the convex hulls for each dice.
        """
        # If no dice are in the image, return nothing
        if self.dice_position.size == 0 or pixelpoints.size == 0:
            return {}
        
        # Choose the most likely dice centres
        centres = self.dice_position[:int(round(self.dice_count)), :2].astype(np.float32)
        
        if labels is None:
            # Using k nearest neighbours, assign each 'dice' pixel to a dice centre
            knn = cv2.ml.KNearest_create()
            
            responses = np.arange(centres.shape[0])
            
            knn.train(centres, cv2.ml.ROW_SAMPLE, responses)
            ret, results, neighbours, dist = knn.findNearest(pixelpoints, 1)
            labels = results.astype(np.int32)
        
        
        pixelpoints = pixelpoints.astype(np.int32)
        
        hull_dict = {i: None for i in range(centres.shape[0])}
        for i in range(centres.shape[0]):
            # The locations of the 'dice' pixels in dice group i
            tmp = pixelpoints[np.nonzero(labels == i)[0], :]
            
            # Create a mask for dice i
            mask = np.zeros((frame_size[0], frame_size[1]), dtype=np.uint8)
            
            # Set all pixels in dice group i to white
            mask[tmp[:, 0], tmp[:, 1]] = 255
            
            # Find the contour which outlines die i
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Join the contours together to merge distoint pieces of the same die
                # Find the convex hull of the resultant contour
                hull_dict[i] = cv2.convexHull(np.concatenate(contours))
            else:
                del hull_dict[i]
            
        return hull_dict
    
    
    def motion_model(self, grey_image):
        """ Using optical flow as a motion model, update the estimated dice centres through dead reckoning.
        """
        
        if self.dice_position.size > 0:
            # Parameters for lucas kanade optical flow
            lk_params = dict( winSize  = (15,15),
                              maxLevel = 2,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            
            dice_centres = self.dice_position[:, 0:2]
            dice_centres = np.flip(dice_centres, 1)
            
            # Update dice positions with motion model guess
            dice_centres_new, st, err = cv2.calcOpticalFlowPyrLK(self.old_grey_image, grey_image, dice_centres, None, **lk_params)
            self.dice_position[:, 0:2] = np.flip(dice_centres_new, 1)
        
        # Add motion model variance
        self.dice_position[:, 2] *= 1.1
        
        # Update reference image
        self.old_grey_image = grey_image.copy()
    
    
    def kalman_update(self, sensors):
        """ Update the Kalman filters with the position estimations from the kmeans clustering.
        """
        
        # Skip update step if the k-means algorithm returned no dice centres
        if sensors.size == 0:
            self.dice_position[:, 2] *= 1.1
            return
        
        
        # Run kNN algorithm to match sensor points found by k-means to known dice centres
        knn = cv2.ml.KNearest_create()
        
        trainData = self.dice_position[:, 0:2]
        responses = np.arange(trainData.shape[0])
        
        knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
        ret, results, neighbours, dist = knn.findNearest(sensors, 1)
        results = results.astype('uint8')
        
        # Init Kalman algorithm values
        initial_variance = 3
        sensor_var = 0.9
        variance_gain = 1.3
        max_distance = 100
        max_var = 5
        
        # If no matching point from sensor, lower weight
        for point_i in set(responses).difference(set(results.flatten())):
            self.dice_position[point_i, 2] *= variance_gain
        
        for i, (x, y) in enumerate(sensors):
            if dist[i] > max_distance:
                # Add new possible point if nearest matching centre is too far away
                new_point = np.array([x, y, initial_variance]).reshape((1, 3)).astype('float32')
                self.dice_position = np.append(self.dice_position, new_point, axis=0)
                
            else:
                # Otherwise, fuse new and old dice points, updating position and variance
                response_i = results[i, 0]
                nearest_point, nearest_point_var = self.dice_position[response_i, :2], self.dice_position[response_i, 2]
                
                K = nearest_point_var / (sensor_var + nearest_point_var) # Kalman gain
                updated_position = K * sensors[i] + (1 - K) * nearest_point
                updated_variance = sensor_var * K
                
                self.dice_position[response_i, :2] = updated_position
                self.dice_position[response_i, 2] = updated_variance
        
        # Delete any dice centres with variance too high
        for i in range(self.dice_position.shape[0] - 1, -1, -1):
            if self.dice_position[i][2] > max_var:
                self.dice_position = np.delete(self.dice_position, i, axis=0)
        
        # Sort dice centres by ascending variance
        self.dice_position = self.dice_position[self.dice_position[:, 2].argsort()]
    
    
    def extract_dice_image(self, frame, hull, size=(50, 50)):
        """ Given an image frame and a die convex hull, return a minimum sized square image containing the die hull
            Return square's centre in frame and image
        """
        
        # Find the minimum sized square which contains the hull
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
        
    
    def find_blobs(self, im):
        """ Return a blob detector for approximate dice objects in frame
        """
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 1500
        params.maxArea = 50000
        
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs.
        return detector.detect(im)
    
    
    def find_dice_count(self, grey_image, thresh):
        """ Estimate the number of dice in frame and update a Kalman filter to track value
        """
        
        # Find the total area occupied by dice in the frame
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = sum([cv2.contourArea(cnt) for cnt in contours])
        
        adjustment_factor = 0.85 # Compensate for the tendency for the contours to give lower
                                 #   areas than the circular blobs
        areas = [(kp.size/2)**2 * np.pi for kp in self.find_blobs(grey_image)]
        
        # Find the estimated average dice size
        if areas:
            average_dice_size = np.average(areas) * adjustment_factor
        else:
            average_dice_size = total_area # If not blobs are found estimate a single die
            
        average_dice_size = min(average_dice_size, # Prevent dice number edge cases:
                                total_area)        #   Impossible to have between 0 and 1 dice
        
        # Estimate number of dice
        try:
            new_dice_count = total_area / average_dice_size
        except ZeroDivisionError:
            new_dice_count = total_area
        
        
        if self.dice_count is None:
            self.dice_count = new_dice_count
        else:
            # Update Kalman filter for number of dice. Uses an experimental equation for Kalman gain to
            #     speed up change when estimate is far from old value
            K = min(0.1 + 0.1 * abs(new_dice_count - self.dice_count), 0.5)
            self.dice_count = K * new_dice_count + (1 - K) * self.dice_count
        
    
    def process(self, frame):    
        """ The main algorithm process. Does basic image processing and calls helper functions.
        """    
        image_processed = frame.copy()
        
        # Preprocess image
        blur = cv2.GaussianBlur(frame, (9,9), 0)
        grey_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        
        # Threshold image and perform morphogical closing.
        ret, thresh = cv2.threshold(grey_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((11, 11), dtype='uint8')
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find the coordinates of pixels belonging to dice
        pixelpoints = np.transpose(np.nonzero(thresh)).astype(np.float32)
        
        # Update the estimated number of dice in frame
        self.find_dice_count(grey_image, thresh)
        
        # Estimate the centres of dice in the frame
        centres = self.k_means_dice(pixelpoints)
        
        if self.dice_position.shape[0] == 0:
            self.dice_position = np.concatenate((centres, np.ones((centres.shape[0], 1))), axis=1).astype('float32')
        
        # Update the positions of the previously estimated dice centres
        self.motion_model(grey_image)
        
        # Update dice centre kalman filters
        self.kalman_update(centres)
        
        # Find the convex hulls which give the shape of each dice
        hull_dict = self.find_dice_hulls(pixelpoints, frame.shape)
        
        for _, hull in hull_dict.items():
            
            hull_centre, hull_bbox, dice_image = self.extract_dice_image(image_processed, hull)
            
            # Draw bounding box around dice
            green = (0, 255, 0)
            cv2.drawContours(image_processed,[hull_bbox],0,green,2)
            
            if RECORD_TRAINING_IMAGES:
                # Save image of dice as a training image
                cv2.imwrite(f'{self.path}\\{self.image_count}.jpg', dice_image)
                self.image_count += 1
            else:
                # Perform classification of each dice using the TensorFlow net
                blob = cv2.dnn.blobFromImage(cv2.cvtColor(dice_image, cv2.COLOR_BGR2GRAY), 1, (50, 50))
                
                self.net.setInput(blob)
                preds = self.net.forward()
                pred_classes = np.argsort(preds[0])[::-1] + 1
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1            
                
                image_processed = cv2.putText(image_processed, str(pred_classes[0]), hull_centre, font, 
                                              fontScale, green, thickness=3)
                
        return image_processed    


class MainApp(App):
    
    def build(self):
        self.widget_inst = CameraWidget()
        
        # schedule app to run at 10 FPS
        Clock.schedule_interval(self.widget_inst.capture_frame, 1/10)
        
        return self.widget_inst
    
    def close_app(self):
        self.widget_inst.close_camera()
        

if __name__ == "__main__":
    app_inst = MainApp()
    app_inst.run()
    app_inst.close_app()
