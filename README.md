# Drowsiness Detector
A Python drowsiness detector based on OpenCV and Dlib.

## Methodology
"Eye Blink Detection using Facial Landmarks"  
obtained from http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf  

The program locates 6 facial landmarks on each eye and computes the "eye aspect ratio".  
The alarms sounds when user's eyes are below a certain eye aspect ratio during a period of time.  

## Dependency
Python 3.4 or higher  
cv2
dlib
imutils
scipy
pygame

## Usage
`python drowsiness_detector.py`  
or  
`python3 drowsiness_detector.py`

## Sources
facial landmark predictor: shape_predictor_68_face_landmarks.dat  
obtained from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.  

This facial landmark predictor model used is trained on the iBUG 300-W face landmark dataset  
from https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/.  
The data set is commercial-use excluded, which indicates this program and related model should not be used commercially without an approval from iBUG authors.  

alarm sound  
obtained from http://sc.chinaz.com/yinxiao/130503066482.htm

imutils  
obtained from https://github.com/jrosebr1/imutils
