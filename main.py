pip install opencv-python
Dlib: Install Dlib using pip:
bash
Copy code
pip install dlib
Numpy: A package for numerical computing, often used alongside OpenCV:
bash
Copy code
pip install numpy
Code Implementation:
Here's a basic example to get started with eye detection using OpenCV and Dlib:

Import necessary libraries:

python
Copy code
import cv2
import dlib
import numpy as np
Initialize Dlib's face detector and facial landmark predictor:

python
Copy code
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
Capture video from the webcam:

python
Copy code
cap = cv2.VideoCapture(0)
Define a function to extract eye regions:

python
Copy code
def get_eye_region(shape):
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    return left_eye, right_eye
Main loop to process each frame:

python
Copy code
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        
        left_eye, right_eye = get_eye_region(shape)
        
        # Draw eye regions
        cv2.polylines(frame, [left_eye], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [right_eye], isClosed=True, color=(0, 255, 0), thickness=2)
    
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
