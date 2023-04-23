from fer import FER
import cv2
import matplotlib.pyplot as plt


cap = cv2.VideoCapture(0)

assert cap.isOpened()

# классификатор распознавания лиц либо OpenCV Haar Cascade, либо MTCNN
model = FER(mtcnn=True)

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # get bounding box
        emotion_box = model.detect_emotions(frameRGB)[0]['box']
        # get dominate emotion and score
        emotion, emotion_score = model.top_emotion(frameRGB)
        # set param
        x, y = emotion_box[0], emotion_box[1]
        w, h = x + emotion_box[2], y + emotion_box[3]
        font = cv2.FONT_HERSHEY_TRIPLEX
        # put emotion
        cv2.putText(frame, emotion, (x, y - 20), font, 1, (0, 0, 255), 3, cv2.LINE_4)
        # put bounding box
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
    except:
        x, y = 10, 10
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(frame, "I can't see you", (x, y - 20), font, 1, (0, 0, 255), 3, cv2.LINE_4)

    # show frame
    cv2.imshow('frame', frame)
    # press q for exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
