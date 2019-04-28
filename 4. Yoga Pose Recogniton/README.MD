# Extended Yoga Mountain Pose Detection

Hey guys and welcome back, in the next few lectures, we are going to dive deeper into how to use Pose Estimation to identify actions in Yoga, correct planking postures and approximate body ratios. We’ll first start off this lecture with Yoga Pose identification, in this case, we shall identify a particular stance called Mountain Pose. If the stance is performed correctly, it will output text to screen of what Pose has been performed. This app can be particularly useful for getting real-time feedback on your exercises whether it may be yoga, tai chi, weight lifting, palates amongst others.

## OpenPose API

Let’s take a look at how we can use OpenPose to achieve this. 
First head over to this link over here

https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md

Go to the heading Keypoint Ordering and you will see the index for each body part that is used by OpenPose. We can tap into any 24 of these body parts including the background whilst they are detectable. You will see how we use this in code a bit later on. This page provides useful information on the pose estimation output and I suggest you read this before we continue.

## Yoga pose detector
Let’s try to understand how we can calculate the angles between various joints to classify a Yoga pose. 
	Mountain pose requires the hands to be stretched upwards. So, we shall calculate the angle using inverse Cosine function.
	To accomplish this we have also find Euclidean distances between left wrist to right wrist and head to wrist.
	Using the Euclidean distance formula: 
	
![alt text](https://lyfat.files.wordpress.com/2012/05/euclidean.png?w=304&h=280)

Python implementation of formula: 

```Distance = math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 )```

We can approximate the Pose to be a mountain stance based on the angle of the hands relative to the neck and shoulder in combination with the closeness of the wrists. 

## Code
So we in our github repo for the course, we’ve included a template for you to code along with, because there are a lot of repetitive code which does not add value to the topic of Pose Estimation. However, we shall instead be focussing on the code require to detect Mountain pose rather than the peripheral code.

Okay so over here in the run_webcam-template.py. You have the basic structure to code our yoga pose detector. I have added in the definitions for the three apps that we shall build. I have already explained earlier what is happening over here.  
We shall place our focus here. When mode 3 is selected we want to run the Yoga Stance Detection app. 
We will first do some distance calculations and then angle calculations. And then below we shall run the mount pose definition. Okay so let’s get cracking!

### Distance Calculations
So wrist distance calculations are simple. We first obtain the Euclidian coordinates of left and right hand wrists relative to the nose.  
the distance between those two points that we are looking at is saved to a variable called m_pose.

### Angle Calculations
We can then calculate two angles, between the neck, elbow and shoulder for both the left and right hands respectively.  You can probably visualize it better from this chart over here. 
![alt text](https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_pose_18.png)

Now we say if mode 3 is selected, and mountain _pose function is true, then we want to draw the action to screen. 
Mountain_pose calculation is defined right up here. We can a add in all our parameters here, angle1, angle 5, head to hand distance for both right and left hands.
If you want you can a debug logger over here.

# Running the app

Okay so that it all we need. Now you can go ahead and run your app on the any video, You can run it on the video provided. You should know how to execute the command by now.  Select 3 for the yoga pose estimation app. And there you have it. You can see as the woman raises here hands and mountain pose is detected accordingly.

```python run_webcam.py --camera video.mp4 ```

Okay so in the next video we’ll continue with this script but add in mode 4 which will be plank detection. Thank you for watching and we’ll see you in the next lecture.
