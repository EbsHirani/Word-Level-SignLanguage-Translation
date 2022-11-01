# Word-Level-SignLanguage-Translation
This project is Word Level Sign Language translator that recognizes words through gestures. We use Posenets to extract key points of the users and then normalize them and pass these keypoints to an LSTM model that is stateless. <br/> <br/>
![alt text](https://github.com/EbsHirani/Word-Level-SignLanguage-Translation/blob/main/images/Flow.jpg)


In addition to this we try other models and configurations and compare the results. <br/>

<img src="https://github.com/EbsHirani/Word-Level-SignLanguage-Translation/blob/main/images/results.jpg" alt="results" width="400"/>
<br/>
We create a pipeline that takes in a video and outputs a sentence and deploy it a Flask API.<br/><br/>

![alt text](https://github.com/EbsHirani/Word-Level-SignLanguage-Translation/blob/main/images/Screenshot%201.jpg)<br/>
![alt text](https://github.com/EbsHirani/Word-Level-SignLanguage-Translation/blob/main/images/Screenshot%202.jpg)
