# -*- coding: utf-8 -*-
import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


# load alarm
mixer.init()
sound1 = mixer.Sound('alarm1.wav')
sound2 = mixer.Sound('alarm2.wav')

# haar cascade files 눈 감지.
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')



lbl=['Close','Open']

# loading model 
model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

# +
while(True):
    # 프레임별로 비디오를 읽고 프레임 변수에 저장
    ret, frame = cap.read()
    # 프레임의 치수를 저장
    height,width = frame.shape[:2] 

    # BRG(파랑, 빨강, 녹색) 이미지를 회색 이미지로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴, left_eye 및 right_eye는 얼굴과 두 눈의 좌표를 저장
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    # 오른쪽 모서리에 검은색 직사각형을 만듭니다.
    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    # 얼굴 주위에 직사각형 만들기
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 1 )
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (0,255,0) , 1 )
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

#    if (rpred[0]==0 and lpred[0]==0):
#        score=score+1
#        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
#    else:
#        score=score-1
#        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    # 점수 계산
    if(rpred[0]==0 and lpred[0]==0):
        score=min(score+1,20)
        cv2.putText(frame,"Closed",(10,height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,'Score:'+str(score),(100,height-20),cv2.FONT_HERSHEY_COMPLEX_SMALL , 1,(255,255,255),1,cv2.LINE_AA)
    
    elif (rpred[0]==1 and lpred[0]==1):
        score=0
        score=max(score-1,0)
        cv2.putText(frame,"Open",(10,height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,'Score:'+str(score),(100,height-20),cv2.FONT_HERSHEY_COMPLEX_SMALL , 1,(255,255,255),1,cv2.LINE_AA)
    elif (rpred[0]==2 and lpred[0]==2):
        score = 0
        cv2.putText(frame,"NO ONE IS PRESENT",(10,height-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    #사람이 졸려 알람을 울립니다
    if score == 2:
        sound1.play()
        
    if(score>4):
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound2.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        #  이것은 빨간색 경계를 추가
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
