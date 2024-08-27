import cv2

#face classifications are to be captured for eg.....nose, ears, eyes etc.
face_cap=cv2.CascadeClassifier("D:/data science/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

video_cap=cv2.VideoCapture(0) #runtime camera is enabled

while True:
    ret,video_data=video_cap.read()
    col=cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
#give black and white color to the image so that it can read all the facial features easily and the color it again

    faces=face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,# specifies how the image size is modified at each image scale
        minNeighbors=3,# how many min neighbor rectangles should be there for capturing images
        minSize=(30,30),# width and height
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for(x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)#(x,y,w,h)=(x axis,y axis,width,height)

    cv2.imshow("video_line",video_data)
    if cv2.waitKey(10) == ord("a"):# when we click a the video will be exited
        break

video_cap.release()