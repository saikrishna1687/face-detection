import cv2
import numpy as np
import os 

def func():
    for i in range(0,75):
        print('-',end='')
    print('\n\t\t\t\tFace Recognisation')
    for i in range(0,75):
        print('-',end='')
    print('\n')
    print('1.photo')
    print('2.video')
    print('3.web cam')
    for i in range(0,75):
        print('-',end='')
    print('\n')
    x=int(input('select the source of image : '))

    if x==1:
        face_detector(input('\nenter the path : '),0)
    elif x==2:
        face_detector(input('\nenter the path : '),10)
    elif x==3:
        face_detector(0,10)

def get_data(str):
    import csv
    with open('data.csv','r') as userFile:
        userFileReader = csv.reader(userFile)
        for row in userFileReader:
            if(row[0]==str):
                for i in range(0,75):
                    print('-',end='')
                print('name ='+row[1])
                print('age ='+row[2])
                print('sex ='+row[3])
                for i in range(0,75):
                    print('-',end='')

def face_detector(src,i):        
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')   
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");


    id = 2 

    cam = cv2.VideoCapture(src)
    cam.set(3, 640) 
    cam.set(4, 480) 

    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    ids=[]

    while True:

        ret, img =cam.read()

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            if (confidence < 50):
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            if id not in ids:
                ids.append(id)
                get_data(str(id))
            
            cv2.putText(img, str(id), (x+5,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.putText(img, str(confidence), (x+w-30,y-5),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))  

        cv2.imshow('camera',img) 

        k = cv2.waitKey(i) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

while 1:
    func()
    p=int(input('Do you want to recognise another face\n\t\t1-Yes\n\t\t0-No\nEnter your choice : '))
    if p:
        func()
    else:
        break
