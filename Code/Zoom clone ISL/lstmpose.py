import mediapipe as mp
import time
import cv2


mapping = {0: 'Adult',1: 'Alright',2: 'Baby',3: 'Boy',
 4: 'Brother',
 5: 'Child',
 6: 'Crowd',
 7: 'Daughter',
 8: 'Family',
 9: 'Father',
 10: 'Friday',
 11: 'Friend',
 12: 'Girl',
 13: 'Grandfather',
 14: 'Grandmother',
 15: 'Hello',
 16: 'Husband',
 17: 'I',
 18: 'King',
 19: 'Man',
 20: 'Monday',
 21: 'Month',
 22: 'Mother',
 23: 'Neighbour',
 24: 'Parent',
 25: 'Player',
 26: 'Pleased',
 27: 'President',
 28: 'Queen',
 29: 'Saturday',
 30: 'Sister',
 31: 'Son',
 32: 'Sunday',
 33: 'Thursday',
 34: 'Today',
 35: 'Tomorrow',
 36: 'Tuesday',
 37: 'Wednesday',
 38: 'Week',
 39: 'Wife',
 40: 'Woman',
 41: 'Year',
 42: 'Yesterday',
 43: 'he',
 44: 'it',
 45: 'she',
 46: 'they',
 47: 'we',
 48: 'you'}

class handDetector():
    def __init__(self, mode=False, maxHands=2,modelC=0, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelC = modelC
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelC,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
 
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(self.results.multi_hand_landmarks)
        # print(len(self.results.multi_hand_landmarks))
        # if self.results.multi_hand_landmarks:
        #     for handLms in self.results.multi_hand_landmarks:
        #         # print("hand", handLms)
        #         if draw:
        #             self.mpDraw.draw_landmarks(img, handLms,)
        #                                       #  self.mpHands.HAND_CONNECTIONS)
                    
        return img,self.results.multi_hand_landmarks 
 
    def findPosition(self, img, handNo=0, draw=True):
 
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 0), cv2.FILLED)
            # myHand = self.results.multi_hand_landmarks[1]
            # for id, lm in enumerate(myHand.landmark):
            #     # print(id, lm)
            #     h, w, c = img.shape
            #     cx, cy = int(lm.x * w), int(lm.y * h)
            #     # print(id, cx, cy)
            #     lmList.append([id, cx, cy])
            #     if draw:
            #         cv2.circle(img, (cx, cy), 2, (0, 0, 255), cv2.FILLED)
 
        return lmList


import numpy as np
class SentencePredictor:
  def __init__(self, interpreter, detector):
    self.interpreter = interpreter
    self.detector = detector
    self.predCount = 0
    self.currPred = ""
    self.isPred = False
    self.input_details = interpreter.get_input_details()
    self.output_details = interpreter.get_output_details()
    # self.frameCount = 0
    self.frames = []
  
  def predict(self, image):
    image, lms = self.detector.findHands(image)
    if lms:
      hand_points = []
      for hands in lms:
        # keypoints= []
        for data_point in hands.landmark:
          hand_points.append([
                data_point.x,
                data_point.y,
                data_point.z,
                ])
        # hand_points.append(keypoints)
      if len(lms) == 1:
          temp_key = [[0,0,0]]*21
          # print("tempppppp: ",temp_key)
          hand_points = hand_points + temp_key
      hand_points = np.array(hand_points)
      rng = hand_points.max(axis = 0).reshape(-1, hand_points.shape[1]) - hand_points.min(axis = 0).reshape(-1, hand_points.shape[1])
      mini = hand_points.min(axis = 0).reshape(-1, hand_points.shape[1])
      # print(rng.shape)
      hand_points = (hand_points - mini)/rng
      hand_points[np.isnan(hand_points)] = 0
      self.frames.append(hand_points)



      if len(self.frames) > 5:
        # print(len(self.frames))
        if len(self.frames) > 50:
          self.frames.pop(0)
        frames = list(self.frames)
        for j in range(len(frames), 50):
          frames.append(np.array([[0,0,0]]*42))
        frames = np.array(frames)
        frames = np.expand_dims(frames, axis = 0)
        frames = frames.reshape(frames.shape[0], frames.shape[1], -1)
        self.interpreter.set_tensor(self.input_details[0]["index"], frames.astype('float32'))
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output_details[0]["index"])
        # print(mapping[np.argmax(pred)], pred[0][np.argmax(pred)])

        # print(pred[0][np.argmax(pred)])
        # print(mapping[np.argmax(pred)])

        if self.isPred:
          if mapping[np.argmax(pred)] == self.currPred:
            self.predCount +=1
            if self.predCount >= 5:
              self.frames = [hand_points]
              self.isPred = False
              self.predCount = 0 
          else:
            self.frames = [hand_points]
            self.isPred = False
            self.predCount = 0

        else:
          if mapping[np.argmax(pred)] == self.currPred:
            self.predCount +=1

          elif pred[0][np.argmax(pred)] >= 0.55:
            self.currPred = mapping[np.argmax(pred)]
            self.predCount = 0

          if self.predCount == 15 and ~self.isPred:
            # print("Found")
            # print("*********************")
            # print(pred[0][np.argmax(pred)])
            print("Word Found:  ", mapping[np.argmax(pred)])
            # self.frames = []
            self.isPred = True
            # self.predCount = 0
            return mapping[np.argmax(pred)]