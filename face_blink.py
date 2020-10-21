import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor =dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font=cv2.FONT_HERSHEY_SIMPLEX

def get_midpoint(p1,p2):
    return int((p2.x+p1.x)/2) , int((p2.y+p1.y)/2)

while(1):
    _,frame =cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        
        landmarks = predictor(gray,face)
        left_point=(landmarks.part(36).x,landmarks.part(36).y) 
        right_point=(landmarks.part(39).x,landmarks.part(39).y) 
        #le_hor_line=cv2.line(frame, left_point,right_point, (0,255,0),2)
                 
        left_point1=(landmarks.part(42).x,landmarks.part(42).y) 
        right_point1=(landmarks.part(45).x,landmarks.part(45).y) 
        #re_hor_line=cv2.line(frame, left_point1,right_point1, (0,255,0),2)
        
        centre_top=get_midpoint(landmarks.part(37), landmarks.part(38))
        centre_bottom=get_midpoint(landmarks.part(41), landmarks.part(40))
        #le_ver_line=cv2.line(frame, centre_top,centre_bottom, (0,255,0),2)
        
        centre_top1=get_midpoint(landmarks.part(43), landmarks.part(44))
        centre_bottom1=get_midpoint(landmarks.part(47), landmarks.part(46))        
        #re_ver_line=cv2.line(frame, centre_top1,centre_bottom1, (0,255,0),2)
        
        ver_line_length = hypot(centre_top[0]-centre_bottom[0], centre_top[1]-centre_bottom[1])
        ver_line_length1 = hypot(centre_top1[0]-centre_bottom1[0], centre_top1[1]-centre_bottom1[1])
        
        #print("1)"+str(ver_line_length)+"\n 2)"+str(ver_line_length1))
        
        hor_line_length = hypot(right_point[0]-left_point[0], right_point[1]-left_point[1])
        hor_line_length1 = hypot(right_point1[0]-left_point1[0], right_point1[1]-left_point1[1])
        
        ratio =(hor_line_length/ver_line_length)
        ratio1=(hor_line_length1/ver_line_length1)
        c_ratio=(ratio+ratio1)/2
        
        #testing  winks from each eye
        #print("left wink rtio: " +str(ratio1))
        #print("right wink rtio: " +str(ratio))
        #print("wink rtio: " +str(c_ratio))
        
        if c_ratio>5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 2, (255,0,255))
            
        #wink testing
        
        #if ((ratio1>4.3) & (ratio<4.3)):
            #cv2.putText(frame, "Left Wink", (50, 150), font, 2, (255,0,255))
        #if ((ratio>4.3) & (ratio1<4.3)):
            #cv2.putText(frame, "Right Wink", (50, 150), font, 2, (255,0,255))  
            
            #Draw all landmarks on detected face
        for point in range(68):
            x=landmarks.part(point).x
            y=landmarks.part(point).y
            cv2.circle(frame, (x,y), 3, (0,0,255),2)
        
        
    cv2.imshow("frame", frame)
    key = cv2.waitKey(27)&0xFF
    if key==27:
        break
    
cap.release()
cv2.destroyAllWindows()