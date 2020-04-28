# Drowsiness Detector
A Python drowsiness detector based on OpenCV and Dlib.

## Methodology
"Eye Blink Detection using Facial Landmarks"  
obtained from http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf  

The program locates 6 facial landmarks on each eyes and compute the "eye aspect ratio".  
The alarms sounds when user's eyes are below a certain eye aspect ratio during a period of time.

## Sources
facial landmark predictor: shape_predictor_68_face_landmarks.dat  
obtained from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.

alarm sound  
from http://sc.chinaz.com/yinxiao/130503066482.htm

imutils  
obtained from https://github.com/jrosebr1/imutils
