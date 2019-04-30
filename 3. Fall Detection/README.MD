# Fall Detection

Hey guys and welcome back. In this lecture I’m going to show you how to perform fall detection using OpenPose. So essentially in fall detection, basic logic is that person’s head goes down and we get drastically change in its head coordinates’ X and Y but mostly in Y axis.
*	So, Firstly the difference between previous and present coordinates is should be positive so neither negative nor zero.
*	We store these differences as an array or buffer to store previous coordinates and compare it with present coordinates. If it greater than our threshold of 25 then fall has been detection.
*	When this happens then, we can print “Fall detected” using OpenCV print function or print it to screen.
With the algorithm in place, lets get started on the implementation.

## Mode Setting
Before we get into fall detection lets make it easy for us to switch between the various app that we shall create. We’ll create a simple case statement to select which app we would like to run. 
So lets start with typing in
   
*	We will declare our mode case statement
*	The first case is a repeat of the last lecture where we counted the amount of humans were detected.

## Coding Fall Detection
*	Case 2 is where we start fall detection.
 *	 So for each human that we detect, we want to get the head point of each of them.
To detect fall we have used y coordinate of head. Coordinates of head in form of normalize form. We convert normalized points to relative point as per the image size.
 y1.append(y) will store y coordinate to compare with previous point. We have used try and except because many time pose estimator can't predict head point
*	human.body_parts[0] is for head point coordinates. We also get the x and y coordinates relative to image

![alt text](https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_pose_18.png)

Then we type in this line which is it's distance between frame and comparing it with threshold value. If this condition is true, then a fall is detected and we can print it to console.

Lets run our app and see what we get.

So as you can see we are successfully detecting fall. Just keep in mind that the quality of the detection is based on the quality of pose estimation. So there may be some false-detections.

## Tweaking the code

Lets tweak the code to make it a bit more practical. First we are going to import OS,  which provides us a way of using operating system dependent functionality. 

We’ll then add in some arguments that will allow us to save the processed video
Now we add in a line that will display the detections to the video window rather than to console. We want the text to display fall Detected whenever the fall occurs.

And then finally we write out the following lines to record the video using the commonly used VideoWrite definition from opencv. If you are familiar with OpenCV, then this code should be standard procedure. 

Now we can run our fall detection again. You can see that it has mode selection for the app that we would like to run. We’ll select 2 for Fall Detection and press enter to run.

Okay and there you have it. You can see that we are adequately detecting a person falling from video. Try this to your own video and see how well it works on your own apps.
 
*	Command to run: ```python run_webcam.py --camera=PATH_TO_VIDEO  --model=cmu’(OPTIONAL) –save_video True```

