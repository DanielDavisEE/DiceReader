__version__ = "0.0.2"

import cv2
import numpy as np

import kivy
#kivy.require('2.0.0')

from kivy.app import App
from kivy.uix.button import Button

from kivy.uix.widget import Widget
from kivy.clock import Clock

from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle
from kivy.uix.image import Image
from kivy.properties import *


class CameraWidget(Widget):
    
    img_frame = ObjectProperty(force_dispatch=True)
    
    def __init__(self, record_training_images=False, **kwargs):
        super().__init__(**kwargs)
        
        self.record_training_images = record_training_images
        
        self.cap = cv2.VideoCapture('./10s_clip.avi')
        _, old_frame = self.cap.read()
        # Preprocess image
        blur = cv2.GaussianBlur(old_frame, (9,9), 0)
        self.old_grey_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        
        ret, frame = self.cap.read()
        
        self.dice_count = None
        self.dice_position = np.ones((0, 3), dtype='float32')
        
        if self.record_training_images:
            dice_type = 'd6'#f'image_folder_{count}'
            face = '6'
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
            self.net = cv2.dnn.readNetFromTensorflow(f'frozen_models\\frozen_graph.pb')
        
        # Init Kivy info
        self.screen_width = frame.shape[1]
        self.screen_height = frame.shape[0]
        self.texture_array = Texture.create(size=(self.screen_width, self.screen_height))
        Window.size = (self.screen_width, self.screen_height)
            
    
    
    def capture_frame(self, *_):
        
        ret, frame = self.cap.read()  # Read an image from the frame.
        if ret:
            
            frame = self.process(frame)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 0)
                
            self.img_frame = self.mute_channels(frame, '')
                    
    
    def on_img_frame(self, *_):
        
        arr = self.img_frame.reshape((-1))
        arr = arr.astype('ubyte')
        
        # now blit the array
        self.texture_array.blit_buffer(arr, colorfmt='rgb', bufferfmt='ubyte')      
        
        with self.canvas:
            Rectangle(texture=self.texture_array, pos=(0, 0), 
                      size=(self.screen_width, self.screen_height))
            
            
    def close_camera(self):
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
        
        compactness, labels, centres = cv2.kmeans(data=pixelpoints,
                                                  K=K,
                                                  bestLabels=bestLabels,
                                                  criteria=criteria,
                                                  attempts=attempts,
                                                  flags=flags,
                                                  centers=centres)
        
        assert centres.shape[1] == 2
        
        return centres
    
    
    def find_dice_hulls(self, pixelpoints, frame_size, labels=None):
        """ Using the dice centres from the kalman posterior values and either a set of cluster labels or a
            thresholded image, find the convex hulls for each dice.
        """
        if self.dice_position.size == 0 or pixelpoints.size == 0:
            return {}
        
        centres = self.dice_position[:int(round(self.dice_count)), :2].astype(np.float32)
        
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
            
        self.dice_position[:, 2] *= 1.1
        
        self.old_grey_image = grey_image.copy()
    
    
    def kalman_update(self, sensors):
        """ Update the Kalman filters with the position estimations from the kmeans clustering.
        """
        
        if sensors.size == 0:
            self.dice_position[:, 2] *= 1.1
            return
        
        knn = cv2.ml.KNearest_create()
        
        # Run kNN algorithm to match sensor points to known points
        trainData = self.dice_position[:, 0:2]
        responses = np.arange(trainData.shape[0])
        
        knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
        ret, results, neighbours, dist = knn.findNearest(sensors, 1)
        results = results.astype('uint8')
        
        initial_variance = 3
        sensor_var = 0.9
        variance_gain = 1.3
        
        # If no matching point from sensor, lower weight
        for point_i in set(responses).difference(set(results.flatten())):
            self.dice_position[point_i, 2] *= variance_gain
        
        for i, (x, y) in enumerate(sensors):
            max_distance = 100
            if dist[i] > max_distance:
                # Add new possible point
                new_point = np.array([x, y, initial_variance]).reshape((1, 3)).astype('float32')
                self.dice_position = np.append(self.dice_position, new_point, axis=0)
                
            else:
                response_i = results[i, 0]
                nearest_point = self.dice_position[response_i]
                K = nearest_point[2] / (sensor_var + nearest_point[2])
                self.dice_position[response_i, :2] = K * sensors[i] + (1 - K) * self.dice_position[response_i, :2]
                self.dice_position[response_i, 2] = sensor_var * K
            
        for i in range(self.dice_position.shape[0] - 1, -1, -1):
            max_var = 5
            if self.dice_position[i][2] > max_var:
                self.dice_position = np.delete(self.dice_position, i, axis=0)
                
        self.dice_position = self.dice_position[self.dice_position[:, 2].argsort()]
    
    
    def extract_dice_image(self, frame, hull, size=(50, 50)):
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
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = sum([cv2.contourArea(cnt) for cnt in contours])
        
        adjustment_factor = 0.85 # Compensate for the tendency for the contours to give lower
                                 #   areas than the circular blobs
        areas = [(kp.size/2)**2 * np.pi for kp in self.find_blobs(grey_image)]
        if areas:
            average_dice_size = np.average(areas) * adjustment_factor
        else:
            average_dice_size = total_area
            
        average_dice_size = min(average_dice_size, # Prevent dice number edge cases:
                                total_area)        #   Impossible to have between 0 and 1 dice
        
        new_dice_count = total_area / max(1, average_dice_size)
        if self.dice_count is None:
            self.dice_count = new_dice_count
        else:
            K = min(0.1 + 0.1 * abs(new_dice_count - self.dice_count), 0.5)
            self.dice_count = K * new_dice_count + (1 - K) * self.dice_count
        
    
    def process(self, frame):        
        image_processed = frame.copy()
        
        # Preprocess image
        blur = cv2.GaussianBlur(frame, (9,9), 0)
        
        # convert the image to grayscale
        grey_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(grey_image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((11, 11), dtype='uint8')
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        pixelpoints = np.transpose(np.nonzero(thresh)).astype(np.float32)
            
        self.find_dice_count(grey_image, thresh)
        
        centres = self.k_means_dice(pixelpoints)
        
        if self.dice_position.shape[0] == 0:
            self.dice_position = np.concatenate((centres, np.ones((centres.shape[0], 1))), axis=1).astype('float32')
        
        self.motion_model(grey_image)
        
        self.kalman_update(centres)
        
        hull_dict = self.find_dice_hulls(pixelpoints, frame.shape)
        
        for _, hull in hull_dict.items():
            
            hull_centre, hull_bbox, dice_image = self.extract_dice_image(image_processed, hull)
            
            # Draw bounding box around dice
            green = (0, 255, 0)
            cv2.drawContours(image_processed,[hull_bbox],0,green,2)
            
            if self.record_training_images:
                cv2.imwrite(f'{self.path}\\{self.image_count}.jpg', dice_image)
                self.image_count += 1
            else:
                # Predict outcomes
                blob = cv2.dnn.blobFromImage(cv2.cvtColor(dice_image, cv2.COLOR_BGR2GRAY), 1, (50, 50))
                
                self.net.setInput(blob)
                preds = self.net.forward()
                
                pred_classes = np.argsort(preds[0])[::-1] + 1
                
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX
                # fontScale
                fontScale = 1            
                
                image_processed = cv2.putText(image_processed, str(pred_classes[0]), hull_centre, font, 
                                              fontScale, green, thickness=3)            
                
            
        #image_processed = cv2.putText(image_processed, str(self.dice_count), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                                      #1, (0, 255, 0), thickness=2)    
        return image_processed    
    
    def gray_to_rgb_format(self, frame):
        new_frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=int)
        old_frame = np.expand_dims(frame, axis=2)
        return new_frame + old_frame
    
    
    def mute_channels(self, frame, channels):
        if channels == '':
            return frame
        
        channel_index = {
            'r': 0,
            'g': 1,
            'b': 2
        }
        channel_selector = np.ones(3, dtype=int)
        for channel in channels:
            channel_selector[channel_index[channel]] = 0
        
        mask = np.zeros(frame.shape, dtype=int) + channel_selector
           
        return frame * mask


class MainApp(App):
    
    def build(self):
        #return Button(text='hello world')
        #cap = cv2.VideoCapture(0)  # Open the first camera connected to the computer.
        #ret, frame = cap.read()  # Read an image from the frame.   
        
        #return Button(text=f'{len(frame)} x {len(frame[0])}')        
        self.widget_inst = CameraWidget()
        
        Clock.schedule_interval(self.widget_inst.capture_frame, 1/10)
        
        return self.widget_inst
    
    def close_app(self):
        self.widget_inst.close_camera()
        

if __name__ == "__main__":
    app_inst = MainApp()
    app_inst.run()
    #app_inst.close_app()
