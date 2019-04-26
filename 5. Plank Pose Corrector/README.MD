# Planking Pose Corrector

Hey guys and welcome back, in the next few lectures, we are going to dive deeper into how to use Pose Estimation to identify actions in Yoga, correct planking postures and approximate body ratios. We’ll first start off this lecture with Yoga Pose identification, in this case, we shall identify a particular stance called Mountain Pose. If the stance is performed correctly, it will output text to screen of what Pose has been performed. This app can be particularly useful for getting real-time feedback on your exercises whether it may be yoga, tai chi, weight lifting, palates amongst others.

## OpenPose API

Let’s take a look at how we can use OpenPose to achieve this. 
First head over to this link over here

https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md

## Plank Detection

So this a continuation of the last lecture where we detected specific yoga stances. The concept is very similar as to yoga pose detection so we shall just focus on coding the Plank Pose corrector.
Lets take a look at how it works.

When planking we have used four angles and two distances to verify the pose. The mention angles and distance are:

1.	Head to left wrist distance (Used points: 0,7)
2.	Head to right wrist distance (Used points: 0,4)
3.	Angle between left shoulder and left wrist (Used points: 7,6,5)
4.	Angle between right shoulder and right wrist (Used points: 4,3,2)
5.	Angle between left hip and left ankle (Used points: 11,12,13)
6.	Angle between right hip and right ankle (Used points: 8,9,10)

![alt text](https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_pose_18.png)

We essentially take all these angles and observe if they enter in the range of our predefined thresholds that represent a plank in real life.

## Coding
We have already have the definition for the plank over here with all the predefined thresholds mentioned earlier. We’ll focus now on the distance and head calculations that will form part of Mode 4 Plank Pose Correction.

So we start of calculating the head to wrist for both the left and right hands. We find the Euclidian distance for the two points relative to the head or rather the nose.

We then attempt to calculate the angles according to the points mentioned earlier which are the angles between the shoulders and wrist.  And also the angles between the hips and ankles for both sides of the body.

So if we detect a pose so length of pose is greater than 0.

If the mode 4 is selected, we can then input these variables into our plank range calculator and returns the action plank if the angles and distances are within our predefined ranges. 

So if true, we will print Plank to screen and debug to console if you want. Simple as that.

Okay so once we have this done, we can go ahead and run this on our video. You can use the video attached to this course or download any video that has a full body view of a person planking. You can observe from the video that at the optimal planking pose is detected quite effectively


# Running the app

Okay so that it all we need. Now you can go ahead and run your app on the any video, You can run it on the video provided. You should know how to execute the command by now.  Select 3 for the yoga pose estimation app. And there you have it. You can see as the woman raises here hands and mountain pose is detected accordingly.

```python run_webcam.py --camera video.mp4 ```

Okay so in the next video we’ll continue with this script but add in mode 4 which will be plank detection. Thank you for watching and we’ll see you in the next lecture.
