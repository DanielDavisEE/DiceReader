# Calibrate camera

# Find dice in image

# Identify type of dice

# Reorientate and crop dice image

# Read value from dice

import cv2

from kivy.app import App
#from kivy.uix.widget import Widget

#from kivy.uix.image import Image
#from kivy.core.window import Window
#from kivy.graphics.texture import Texture
#from kivy.graphics import Rectangle

from kivy.uix.button import Button

from array import array


class MainApp(App):
    def build(self):
        
        #texture = Texture.create(size=(640, 480), colorfmt='rgb')
        
        cap = cv2.VideoCapture(0)  # Open the first camera connected to the computer.
        ret, frame = cap.read()  # Read an image from the frame.   
        
        return Button(text=f'{len(frame)} x {len(frame[0])}')
    '''
        # Release the camera device and close the GUI.
        cap.release()
        
        #cv2.imshow('frame', frame)  # Show the image on the display.
        # 480 * 640
        #print(ret)
        #print(len(frame), len(frame[0]))
        
        arr = array('B', [val for row in frame for col in row for val in col])
        
        # now blit the array
        texture.blit_buffer(arr, colorfmt='rgb', bufferfmt='ubyte')      
        
        bg = Widget()
        with bg.canvas:
            Rectangle(texture=texture, pos=(0, 0), size=(640, 480))
        
        return bg
'''


if __name__ == "__main__":
    MainApp().run()
