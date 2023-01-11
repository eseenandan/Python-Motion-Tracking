import cv2 as cv 
import mediapipe as mp 
import time 


class handsDetector:
    # i think passing values in here means by default 
    def __init__(self,mode = False, maxHands = 2, modelC = 1, detectionCon = 0.5, trackingCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        
        self.mpHands = mp.solutions.hands
        # control click on function to see how the function works
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelC, self.detectionCon, self.trackingCon) # hands is going to take params this time
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # i guess you can declare a variable with self and use it in other methods as well
        self.results = self.hands.process(imgRGB)
        
        
        # for testing to see if theres hands
        # print(results.multi_hand_landmarks) 
        
        if self.results.multi_hand_landmarks:
            for handLandMarks in self.results.multi_hand_landmarks: 
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandMarks, self.mpHands.HAND_CONNECTIONS)
        
        return img
    
    def findPosition(self, img, handNum = 0, draw = True):
        landmarkList = []
        
        if self.results.multi_hand_landmarks:
            
            # determine what hand youre talking about 
            myHand = self.results.multi_hand_landmarks[handNum]
            # get cordinates of that hand
            for id, landmark in enumerate(myHand.landmark):
                    print(id,landmark)
                    # gets width and height 
                    height, width, channel = img.shape
                    
                    # position of x and y centers 
                    cx, cy = int(landmark.x*width), int(landmark.y*height)
                    # print(id, cx, cy)
                    landmarkList.append([id,cx,cy])
                    # enlarges a circle at the ID position on the hand (those little dots are the IDs)
                    if draw:
                        cv.circle(img, (cx,cy), 15, (0,0,200), cv.FILLED)
                        
        return landmarkList

def main():
    prevTime = 0
    curTime = 0
    capture = cv.VideoCapture(0)
    detector = handsDetector() # uses default params
    # Code used to run the webcam
    while True:
        success, img = capture.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4]) # tip of thumb position
        # FPS
        curTime = time.time()
        fps = 1 / (curTime-prevTime)
        prevTime = curTime

        # display fps on screen
        cv.putText(img, str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 3)

        # open results and extract info within
        resized = cv.resize(img, (1000,500), interpolation = cv.INTER_AREA)
        cv.imshow("Image", resized)
        cv.waitKey(1)


    

if __name__ == "__main__":
    main()
    