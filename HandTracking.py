import cv2 as cv 
import mediapipe as mp 
import time 

capture = cv.VideoCapture(0)

mpHands = mp.solutions.hands
# control click on function to see how the function works
hands = mpHands.Hands() 
mpDraw = mp.solutions.drawing_utils

prevTime = 0
curTime = 0

# Code used to run the webcam
while True:
    success, img = capture.read()
    # convert image to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    
    # for testing to see if theres hands
    # print(results.multi_hand_landmarks) 
    
    if results.multi_hand_landmarks:
        for handLandMarks in results.multi_hand_landmarks: 
            # check index number of each finger
            for id, landmark in enumerate(handLandMarks.landmark):
                print(id,landmark)
                # gets width and height 
                height, width, channel = img.shape
                
                # position of x and y centers 
                cx, cy = int(landmark.x*width), int(landmark.y*height)
                print(id, cx, cy)
                
                # enlarges a circle at the ID position on the hand (those little dots are the IDs)
                if id == 0:
                    cv.circle(img, (cx,cy), 75, (200,5,100), cv.FILLED)
                
            mpDraw.draw_landmarks(img, handLandMarks, mpHands.HAND_CONNECTIONS)
    
    # FPS
    curTime = time.time()
    fps = 1 / (curTime-prevTime)
    prevTime = curTime
    
    # display fps on screen
    cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 3)
    
    # open results and extract info within
    resized = cv.resize(img, (500,500), interpolation = cv.INTER_AREA)
    cv.imshow("Image", resized)
    cv.waitKey(1)
    