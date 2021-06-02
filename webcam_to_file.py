# webcam_to_file.py

import cv2

# Open the first camera connected to the computer.
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()  # Read an frame from the webcam.

    out.write(frame)  # Write the frame to the output file.

    cv2.imshow('frame', frame)  # While we're here, we might as well show it on the screen.

    # Close the script when q is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera device and output file, and close the GUI.
cap.release()
out.release()
cv2.destroyAllWindows()
