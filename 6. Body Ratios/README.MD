# Body Ratio Calculation

This lecture is a continuation of the previous 2 lecture where we use pose estimation to perform activity recognition. In this lecture we shall be calculating the leg to body ratios which is a very simple equation.
Defined as the ratio of total leg length to total height

LBR = Leg length/Total Height.

![alt text](https://www.acsh.org/sites/default/files/menlonglegs3.jpg)

Now of course, there are numerous body ratios that you can calculate, but for now we’ll just focus on LBR as the concept is the same for all the other ratios and proportions. 

![alt text](https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/doc/media/keypoints_pose_18.png)

## Code
Unlike the previous lecture where we used angles for our calculations. For this lecture we will just use the Euclidean distance throughout. 
We then calculate the leg to body ratio which is a simple average for both left and right sides and then division to the total height of the person. 

[CODING]

We just need one side to calculate the ratio however if you want to get an average of the two sides you can do so.
Once we have the calculations of the ratio in place for the average LBR. We can then proceed to print the LBR text to screen. Simple line of code as we have done before.

## Running the Code
Okay once that is completed, Let us run the code, Type in 
Python webcam.py –camera and the path to our video. Select mode 5 for our Body Ratio Calculator and hit enter to run. 
And there you have it.  We are successfully able to calculate the Leg to body ratio in real time
Now if you want to be creative and go beyond this lesson, I suggest you, try calculating other body ratios out and see what you output it is. To go beyond that, see if you can do it for multiple people simultaneously.

## Wrap up
So whether or not you believe in the study mentioned earlier, you can rest assure that no matter what your LBR is, you can make a difference in the world with your awesome skills in AI. 

Thank you for watching and we’ll see you in the next lecture.
