import numpy as np
import cv2
import os

AGE_BUCKETS=["(0-2)","(4-6)","(8-12)","(15-20)","(25-32)",
             "(38-43)","(48-53)","(60-100)"]
# memuat pendeteksian wajah
prototxtPath=("face_detector/deploy.prototxt")
weightsPath=("face_detector/res10_300x300_ssd_iter_140000.caffemodel")

faceNet=cv2.dnn.readNet(prototxtPath,weightsPath)

prototxtPath=("age_detector/age_deploy.prototxt")
weightsPath=("age_detector/age_net.caffemodel")
ageNet= cv2.dnn.readNet (prototxtPath, weightsPath)

# load input image dan construct
image=cv2.imread('61.jpg')
(h, w)=image.shape[:2]
blob=cv2.dnn.blobFromImage(image,1.0,(30,300),
     (104.0, 177.0, 123.0))

# pass the bloob
print ("[info] computing face detection...")
faceNet.setInput (blob)
detections=faceNet.forward()

# looping over detections
for i in range (0,detections.shape[2]):

    confidence=detections[0,0,i,2]

    if confidence > 0.5:
        box=detections[0,0,i,3:7]*np .array([w,h,w,h])
        (startX,startY,endX,endY)=box.astype("int")


        face=image[startX:endX]
        faceBlob=cv2.dnn.blobFromImage(face,1.0,(227,227),
           (78.4263377603,87.7689143744,144.895847746),
        swapRB=False)

        ageNet.setInput(faceBlob)
        preds=ageNet.forward()
        i=preds[0].argmax()
        age=AGE_BUCKETS[i]
        ageConfidence=preds[0][i]

        text="{}:{:.2f}%".format(age,ageConfidence * 100)
        print("[info]{}".format(text))

        y=startY-10 if startY -10 > 10 else startY + 10
        cv2.rectangle(image,(startX,startY),(endX,endY),
            (0,0,255),2)
        cv2.putText(image,text,(startX,y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,255),2)
cv2.imshow("Image",image)
cv2.waitKey(0)





