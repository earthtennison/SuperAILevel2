# Gaming Posture Detector
First, I want to thank to illdonet who provide valuable posture estimation code.<br>
In this project, I have modified some code from [tf-pose-estimation](https://github.com/gsethi2409/tf-pose-estimation)

##Step by step
1. Build new python environment (python 3.8)
3. Clone the repository [tf-pose-estimation](https://github.com/gsethi2409/tf-pose-estimation)
4. Copy armDetector.py from my Github to that environment
5. replace estiator.py to the old one tf-pose-estimation -> tf-pose -> estimator.py\
In estimator.py, I have created new function draw_arm, getLength, getAngle

![](https://github.com/earthtennison/SuperAILevel2/blob/main/GamingPostureDetector/Screenshot%202021-02-23%20015156.png)\
![](https://github.com/earthtennison/SuperAILevel2/blob/main/GamingPostureDetector/Screenshot%202021-02-23%20015222.png)
