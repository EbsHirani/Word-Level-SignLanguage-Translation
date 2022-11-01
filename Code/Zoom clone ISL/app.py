import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, abort
from twilio.jwt.access_token import AccessToken
from twilio.jwt.access_token.grants import VideoGrant, ChatGrant
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from io import StringIO
import io
import base64
from PIL import Image
import cv2
import numpy as np
import imutils
import tensorflow as tf

from flask_socketio import SocketIO, emit
from keras.models import Model
from keras.layers import Dense, LSTM, Input

from lstmpose import SentencePredictor, handDetector

load_dotenv()
twilio_account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
twilio_api_key_sid = os.environ.get('TWILIO_API_KEY_SID')
twilio_api_key_secret = os.environ.get('TWILIO_API_KEY_SECRET')
twilio_client = Client(twilio_api_key_sid, twilio_api_key_secret,
                       twilio_account_sid)

UPLOAD_FOLDER = 'data/'
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_chatroom(name):
    for conversation in twilio_client.conversations.conversations.stream():
        if conversation.friendly_name == name:
            return conversation

    # a conversation with the given name does not exist ==> create a new one
    return twilio_client.conversations.conversations.create(
        friendly_name=name)


# LR = 1e-2
# # inp = Input(shape = (None, X.shape[2]))
# inp = Input(batch_shape = (1, None, 126))

# lstm1 = LSTM(128, dropout=0.2, return_sequences = True, stateful = True)
# lstmin =lstm1(inp)
# lstm2 = LSTM(128, dropout=0.2, stateful= True)
# lstmout = lstm2(lstmin)
# # lstmout = LSTM(64, )(lstmout)
# dense = Dense(49 ,activation = "softmax")
# out = dense(lstmout)
# model = Model(inp, out)
# model.load_weights("lstm_weights.best.hdf5")
# model.reset_states()
# LstmModel = Model(inp, lstmout)
# softModel = Model(lstmout, out)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()


predictor = SentencePredictor(interpreter,handDetector())


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    print("Shubhammmmmmm")
    # print(request.json)
    # print("whatttt ", request.get_json(force=True, silent=True, cache=False))
    username = request.get_json(force=True).get('username')
    print("Shubhammmmmmm ", username)
    if not username:
        abort(401)

    conversation = get_chatroom('My Room')
    try:
        conversation.participants.create(identity=username)
    except TwilioRestException as exc:
        # do not error if the user is already in the conversation
        if exc.status != 409:
            raise

    token = AccessToken(twilio_account_sid, twilio_api_key_sid,
                        twilio_api_key_secret, identity=username)
    token.add_grant(VideoGrant(room='My Room'))
    token.add_grant(ChatGrant(service_sid=conversation.chat_service_sid))

    return {'token': token.to_jwt().decode(),
            'conversation_sid': conversation.sid}

@socketio.on('image')
def image(data_image):
    headers, data_image = data_image.split(',', 1) 
    # print(data_image)
    sbuf = StringIO()
    sbuf.write(data_image)

    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    cv2.imwrite("shubham.jpg",frame)
    print(frame.shape)
    ret = predictor.predict(frame)
    if ret!=None:
        print("PREDDD: ",ret)
        emit('response_back_text', stringData)
        

    # Process the image frame
    # frame = imutils.resize(frame, width=700)
    # frame = cv2.flip(frame, 1)
    imgencode = cv2.imencode('.jpg', frame)[1]

    # im = imgencode.convert('RGB')
    # cv2.imwrite()
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)

if __name__ == '__main__':
    # app.run(host='0.0.0.0')
    socketio.run(app, host='127.0.0.1')
    # socketio.run(app)
