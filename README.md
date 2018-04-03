## Introduction
When I took an machine learning class back in school which talks about this method called Naive bayes classifer. I have no idea how to actually apply it except accepting and move on with the bayes equation. I didnt actually thought of applying it until I started working on perception system of autonomous vehicles, where I applied it to determine if there is an oncoming vehicle at T junctions. It turn out to be surprisingly useful and intuitive.

<p align="center">
    <img src="https://zero2ml.com/img/post/motiondetector-bayes-opencv/motion_detected.gif">
</p>

This post will cover how a bayes inference can be useful to build a consistent motion detection system in a video or camera feed. To get started, you need to have opencv with python binding installed on your system. If you havent, please check out my previous post [here](https://zero2ml.com/post/3-ubuntu-opencv-python-cuda/).

## **Background Subtraction**
An important element in motion detection using opencv is background subtraction. The primary method for motion detection is to use a [**Gaussian Mixture Model-based foreground**](http://www.ee.surrey.ac.uk/CVSSP/Publications/papers/KaewTraKulPong-AVBS01.pdf) function. There is also another method known as [background segmentation](http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf). I have tried both methods for this post and found that the former works better than the latter.

- Gaussian Mixture Model-based foreground:  ***[cv2.BackgroundSubtractorMOG](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_video/py_bg_subtraction/py_bg_subtraction.html)*** in opencv
- Background Segmentation: ***cv2.BackgroundSubtractorMOG2*** in opencv

Bulk of the assumption here is that the background of the video stream is static (i.e a stationary camera) and does not change much from frame to frame. As such, we can model and monitor the background, detecting significant changes which usually correspond to motion in the video.

In the real world, there are changes of lighting and shadow movement which contributes to the error in the motion abstracted from background subtraction method. *cv2.BackgroundSubtractorMOG2* method tend to preserve shadows which probably explains a poor performance here.

## **Problem Formulation**
In reality, the background subtraction isnt perfect which results in failing to detect motion in some frames. This causes disruption in detection consistency which probably an annoyance since our detector is "not sure enough" in deciding if there is motion or not. We need to fix this.

### **Bayesian inference**
Bayesian inference is a statisical method in machine learning, in which [**Bayes' theorem**](https://en.wikipedia.org/wiki/Bayes%27_theorem) is applied to update the probability of a hypothesis as more evidence became available. It is useful for analyzing a sequence of data which is particularly applicable to our motion detector case. Bayesian inference returns the posterior probability as directly proportional to prior probability and likelihood function. The latter parameters are usually pre-determined from observed data (i.e frequency table). The posterior probability would then give us a confidence value in determining the outcome of the situation.

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|\neg A)P(\neg A)} = \frac{P(B|A)P(A)}{P(B)}
$$

where,

- *P(A|B)* ~ Posterior probability
- *P(B|A)* ~ Likelihood
- *P(A)*   ~ Prior probability
- *P(B)*   ~ Evidence

### **Model**
Now, picture a set of detection results captured by the sensor through a series of video frames between time *t = 0* to *t = T*, where *T* is the current time. Each result returned is 1 if the detector picks up motion and 0 otherwise:

$$
	setOfDetections = [1,1,0,1,0,0,1,1,1,....,0,0,0,1]
$$

Base on this, we can assume that each detection instance is independent of each other (why? because why not! Each observation is done on an isolated frame which is discrete in nature). This is an important assumption for naive bayes algorithm to work.

Let *H* be the event that there is motion in the scene and *O* be the event that a motion is detected by the sensor. These two events are different, the former is real, the latter is a measurement from the sensor. Then we have *P(O)* as the evidence of detecting a motion over a significant number of frames and *P(H)* as the prior probability of a motion happening in the frame.

So, lets set some values to parameters of interest:

- *P(H)*    ~ set to 0.5 (probability of motion set to 50%)
- *P(O|H)*  ~ set to 0.8 (probability of observed motion given there is in actual fact motion)
- *P(O|~H)* ~ set to 0.1 (probability of observed motion given there is in actual fact no motion)
- *P(~O|H)* ~ set to 0.1 (probability of no observed motion given there is in actual fact motion)
- *P(~O|~H)* ~ set to 0.8 (probability of no observed motion given there is in actual fact no motion)

In actual fact, those parameters above can be "learnt" through a frequency table if your sample size is large. But for now, we just estimate those values based for simplicity sake.

Recall the Bayes Theorem, we have the following for the first frame:

$$
	P(H|O_1) = \frac{P(O_1|H)P(H)}{P(O_1|H)P(H) + P(O_1|\neg H)P(\neg H)}
$$

subsequently the next frame, using the posterior from the previous result:

$$
	P(H|O_1 \cap O_2) = \frac{P(O_2|H)P(H|O_1)}{P(O_1|H)P(H|O_1) + P(O_1|\neg H)P(\neg H|O_1)}
$$

And so on ...

In the end, we will have a confidence value at *n-th* frame like the following:

$$
	P(H|O_1 \cap O_2 \cap O_3 ... \cap O_n)
$$ 

Which is "confident" enough to estimate the outcome.

## **Remarks**
This [code snippet](https://github.com/khaixcore/motion_detector_bayes) is also used in my diy home surveillance app, [DeepEye](https://github.com/khaixcore/DeepEye). It is a complete python app that allow one to monitor his or her home using a connected USB camera. It also send an email upon detection of activities such as motion, pets and human. Here are some of screenshots in action!

Motion Detection 1         |  Motion Detection 2             
:-------------------------:|:-------------------------:
![](https://khaixcore.github.io/img/project/deepcam/screenshot_5.png)  |  ![](https://khaixcore.github.io/img/project/deepcam/screenshot_6.png)


