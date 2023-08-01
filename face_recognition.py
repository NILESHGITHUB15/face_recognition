import face_recognition
import cv2
import numpy as np
import sys
from threading import Thread

all_location = []
class face_reco:
    def __init__(self,image) -> None:
        self.encode_image = image
        self.all_location = []
    
    def _sort(self,arr):
        res = []
        arr = sorted(arr,key=lambda x:x[0])
        last = ""
        for ar in arr:
            if ar[1] == last:
                continue
            last = ar[1]
            res.append(ar[1])
        return res

    def searching_video(self,video):
        cap = cv2.VideoCapture(video)
        res = []
        time = 0
        while True:
            success,frame = cap.read()
            if not success:break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            faces_frame = face_recognition.face_locations(frame)
            encode = face_recognition.face_encodings(frame,faces_frame)
            if cv2.waitKey(1) == ord("q"):break
            for enc,position in zip(encode,faces_frame):
                t = face_recognition.compare_faces([self.encode_image],enc)
                cv2.rectangle(frame,(position[3],position[0]),(position[1],position[2]),(255,0,255),2)
                if t[0]:
                    cv2.putText(frame,"True",(position[3],position[0]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),4,cv2.LINE_AA)
                    print(video,end=" ")
                    res.append([time,video])
            cv2.imshow(video,frame)
            time+=1
        all_location.append(res.copy())

def main():
    if len(sys.argv) <= 2:
        print("usage: python3 main.py image video1 video2....")
        sys.exit()
    image = sys.argv[1]
    image = cv2.imread(sys.argv[1])
    video = sys.argv[2:]
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    li = []
    for vid in video:
        li.append(face_reco(encode))
        t = Thread(target=li[len(li)-1].searching_video,args=(vid,))
        t.start()
    print()    

if __name__ == "__main__":
    main()