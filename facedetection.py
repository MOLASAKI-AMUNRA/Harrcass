import cv2
alg="haarcascade_frontalface_default.xml"
harcas=cv2.CascadeClassifier(alg)
cam =cv2.VideoCapture(0)
while True:
    ret,img=cam.read()
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=harcas.detectMultiScale(grayImg,1.3,4)
    for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("FaceDetection",img)
        key=cv2.waitKey(1)
        if key==27:
            break
cam.release()
cv2.destroyAllWindows()


