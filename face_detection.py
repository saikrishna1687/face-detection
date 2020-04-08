import cv2
import os

def func():
    for i in range(0,75):
        print('-',end='')
    print('\n\t\t\t\tenter details')
    for i in range(0,75):
        print('-',end='')
    face_id=get_details()
    print('1.photo')
    print('2.video')
    print('3.web cam')
    x=int(input('select the source of image'))

    if x==1:
        for i in range(0,5):
            face_detector(input('enter the path'),face_id)
    elif x==2:
        face_detector(input('enter the path : '),face_id)
    elif x==3:
        face_detector(0,face_id)

def get_details():
    import csv
    with open('data.csv', mode='a') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        face_id=input('Enter the unique id : ')
        name=input('enter name : ')
        age=input('enter age : ')
        sex=input('enter gender : ')
        hei=input("enter height : ")
        writer.writerow([face_id,name,age,sex,hei])
        for i in range(0,75):
            print('-',end='')
    return face_id

def face_detector(src,face_id):
    cam = cv2.VideoCapture(src)
    cam.set(3, 640)
    cam.set(4, 480)

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    print("\n [INFO] Initializing face capture.")
    count = 0

    while count < 30: 

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff 
        if k == 27:
            break

    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

while 1:
    func()
    if int(input('do you want to give another face')):
        func()
    else:
        import face_training
        break
