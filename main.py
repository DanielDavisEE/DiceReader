__version__ = "0.0.2"

#import cv2
#import numpy as np

import kivy
#kivy.require('2.0.0')

from kivy.app import App
from kivy.uix.button import Button

#from kivy.uix.widget import Widget
#from kivy.clock import Clock

#from kivy.uix.image import Image
#from kivy.core.window import Window
#from kivy.graphics.texture import Texture
#from kivy.graphics import Rectangle
#from kivy.uix.image import Image
#from kivy.properties import *

#from array import array

"""
class CameraWidget(Widget):
    
    img_frame = ObjectProperty(force_dispatch=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        
        self.screen_width = frame.shape[1]
        self.screen_height = frame.shape[0]
        self.texture_array = Texture.create(size=(self.screen_width, self.screen_height))
        Window.size = (self.screen_width, self.screen_height)
    
    
    def capture_frame(self, *_):
        
        ret, frame = self.cap.read()  # Read an image from the frame.
        if ret:
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frame = self.process(frame)
            
            #filename = 'Pic.png'
            #cv2.imwrite(filename, frame)
            #with self.canvas:
                #clear()
                #Image(source=filename)
                
            self.img_frame = self.mute_channels(frame, '')

    
    def process(self, frame):
        
        blur = cv2.GaussianBlur(frame, (9,9), 0)
        img_data = blur / 255.0
        img_data = img_data.reshape((-1, 3))
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        img_data = img_data.astype(np.float32)
        compactness, labels, centers = cv2.kmeans(img_data,
                                                  8, None, criteria,
                                                  10, flags)
        
        new_colors = centers[labels].reshape((-1, 3))
        image_recolored = new_colors.reshape(frame.shape) * 255
        image_recolored = image_recolored.astype('uint8')
        
        # convert the image to grayscale
        gray_image = cv2.cvtColor(image_recolored, cv2.COLOR_BGR2GRAY)
        
        # convert the grayscale image to binary image
        ret, thresh = cv2.threshold(gray_image, 127, 255, 0)           
        
        # find contours in the binary image
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_recolored, contours, 0, (0,255,0), hierarchy=hierarchy, maxLevel=1)
        
        #for c in contours:
            ## calculate moments for each contour
            #M = cv2.moments(c)
            
            ## calculate x,y coordinate of center
            #if M["m00"] != 0:
                #cX = int(M["m10"] / M["m00"])
                #cY = int(M["m01"] / M["m00"])
            #else:
                #cX, cY = 0, 0
            #cv2.circle(image_recolored, (cX, cY), 5, (255, 255, 255), -1)
        
        #imgray = cv2.cvtColor(image_recolored, cv2.COLOR_BGR2GRAY)
        #ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cnt = contours[4]
        
        
        ''' 
        blur = cv2.GaussianBlur(frame, (9,9), 0)
        input_image  = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
        
        # Threshold your image to make sure that is binary
        #thresh_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        #th3 = cv2.adaptiveThreshold(input_image, 255, thresh_type, \
                    #cv2.THRESH_BINARY, 11, 1)        
        
        # Threshold your image to make sure that is binary
        binary_image = cv2.adaptiveThreshold(input_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 3)        
        
        #binary_image_cleaned = cv2.fastNlMeansDenoising(binary_image, None, 10, 7, 21)
        
        # Perform connected component labeling
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image,
                                                                              connectivity=4)
        # Create false color image
        colors = np.random.randint(0, 255, size=(n_labels , 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # for cosmetic reason we want the background black
        false_colors = colors[labels]
        
        #cv2.imshow('binary', binary_image)
        #cv2.imshow('false_colors', false_colors)
        
        false_colors_draw = false_colors.copy()
        for centroid in centroids:
            cv2.drawMarker(false_colors_draw, (int(centroid[0]), int(centroid[1])),
                           color=(255, 255, 255), markerType=cv2.MARKER_CROSS)
        #cv2.imshow('false_colors_centroids', false_colors_draw)
        
        return false_colors#self.gray_to_rgb_format(binary_image)'''
        
        return image_recolored
    
    
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
 """   


class MainApp(App):
    
    def build(self):
        return Button(text='hello world')
        #cap = cv2.VideoCapture(0)  # Open the first camera connected to the computer.
        #ret, frame = cap.read()  # Read an image from the frame.   
        
        #return Button(text=f'{len(frame)} x {len(frame[0])}')        
        #self.widget_inst = CameraWidget()
        
        #Clock.schedule_interval(self.widget_inst.capture_frame, 1/10)
        
        #return self.widget_inst
    
    #def close_app(self):
        #self.widget_inst.close_camera()
        

if __name__ == "__main__":
    app_inst = MainApp()
    app_inst.run()
    #app_inst.close_app()
