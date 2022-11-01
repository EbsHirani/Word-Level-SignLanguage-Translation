import tflearn
import mediapipe as mp
import cv2, pickle
import numpy as np
from parrot import Parrot
# import torch
import warnings


warnings.filterwarnings("ignore")

''' 
uncomment to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)
'''
X_shape = (15166, 109, 126)
#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")


# mapping = pickle.load("")
mapping = dict({0: 'Alright',1: 'Baby',2: 'Boy',3: 'Brother', 4: 'Daughter', 5: 'Family', 6: 'Father', 7: 'Friday', 8: 'Grandfather',9: 'Grandmother', 10: 'Hello', 11: 'Husband', 12: 'I', 13: 'King', 14: 'Man', 15: 'Monday', 16: 'Month', 17: 'Mother',18: 'Neighbour', 19: 'Parent', 20: 'Pleased', 21: 'President', 22: 'Queen', 23: 'Saturday', 24: 'Sister', 25: 'Son',26: 'Sunday',27: 'Thursday', 28: 'Today', 29: 'Tomorrow', 30: 'Tuesday', 31: 'Wednesday', 32: 'Week', 33: 'Wife', 34: 'Woman', 35: 'Year',36: 'Yesterday', 37: 'he', 38: 'it', 39: 'she', 40: 'they', 41: 'we', 42: 'you'})

net = tflearn.input_data(shape=[None, 109, 126])
net = tflearn.lstm(net, 128, dropout=0.8, return_seq=True)
net = tflearn.lstm(net, 128)
net = tflearn.fully_connected(net, 43, activation='softmax')
net = tflearn.regression(net, optimizer='adam',
                          loss='categorical_crossentropy', name="output1")

model = tflearn.DNN(net, tensorboard_verbose=0)
model.load("tflearn/model_4.tflearn")


class handDetector():
    def __init__(self, mode=False, maxHands=2,modelC=1, detectionCon=0.5, trackCon=0.5):
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
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # print("hand", handLms)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,)
                                              #  self.mpHands.HAND_CONNECTIONS)
                    
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

class SentencePredictor:
  def __init__(self, model, detector):
    self.model = model
    self.detector = detector
    self.predCount = 0
    self.currPred = ""
    self.isPred = False
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



      if len(self.frames) > 20:
        # print(len(self.frames))
        if len(self.frames) > 109:
          self.frames.pop(0)
        frames = list(self.frames)
        for j in range(len(frames), X_shape[1]):
          frames.append(np.array([[0,0,0]]*42))

        frames = np.array(frames)
        frames = np.expand_dims(frames, axis = 0)
        frames = frames.reshape(frames.shape[0], frames.shape[1], -1)
        pred = model.predict(frames)
        # print(pred[0][np.argmax(pred)])
        # print(mapping[np.argmax(pred)])

        if self.isPred:
          if mapping[np.argmax(pred)] == self.currPred:
            self.predCount +=1
            if self.predCount >= 25:
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

          elif pred[0][np.argmax(pred)] >= 0.8:
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
        # print("here")



def getSentence(path):
    predictor = SentencePredictor(model, handDetector())
    phrase = ""
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    
    while success:
        # print(count)
        count += 1
        ret = predictor.predict(image)
        if ret != None:
            phrase += ret + " "
        success,image = vidcap.read()
    
    para_phrases = parrot.augment(input_phrase=phrase, use_gpu=False)
    try:
        return para_phrases[0][0]
    except:
        return phrase