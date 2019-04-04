# Installation Guide Of Pose Estimation

Hey guys and welcome back in lecture I’m going to take through the installation of pose estimation step-by-step. What we will achieve by the end of this tutorial will be us running pose estimation on a video. Okay so lets get straight into it. 

## Downloading the repo:

1.	Okay first we go to the Pose Estimation github page, we just type in tf pose estimation github. Or click the link attached to the course
	*	We will click on this link ildoonet
	*	And then simply click to download the zip or if you want to be cool, you can do a git clone in anaconda prompt. 
	*	Then lets create a folder where we can work from and call it pose estimation.
	*	Let be cool and download it via git clone. Press enter and let it do its thing. https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose

2.	In the meantime we Download swig which is a tool that connects programs written in c and c++ with high level programming languages which in this case we are using python. 
	*	Type is swig for windows and click on the first link which say Getting Started on Windows – Swig
	*	Scroll down to 3.1 Installation on windows and click on Swig website link.
	*	Click download for the latest version
	*	From the check boxes select python and windows – submit
	*	Download area and then finally  swigwin-3.0.12 for windows. I’l have the direct link attached to this video. http://www.swig.org/download.html
	
			
3.	Once it has downloaded we can then copy and paste swig into our pose estimation workspace and Extract all the files.
4.	Go to “Edit system environment variables” and add the path of extracted swig folder. 
	*	Click new and paste the folder directory
	*	Click okay to exit 
5.	Back in the root folder for pose estimation, you open up cmd. 
	*	Install requirements using this command ``` install –r requirements.txt```  to install all our dependencies 
6.	Next copy and paste the following line ``` cd tf_pose/pafprocess``` and then paste in  ``` swig –python –c++ pafprocess.i && python setup.py build_ext --inplace```  using traditional python instead of python 3

## Downloading the model:

7.	To get the CMU model, we can navigate through the folders to find this file and open it up with notepad ++ or a text editor of your choice. 
	*	We will copy this link and paste it into our browser
	*	Then click to Download CMU model graph using this [Link] http://www.mediafire.com/file/qlzzr20mpocnpa3/graph_opt.pb
8.	Now returning to the pose estimation folder and then go to \models\graph\cmu and copy downloaded model here.
9.	In cmd, ensure that you cd into the root pose estimation directory
10.	Open file ``` run_webcam.py``` and edit in line no.24 parser.add_argument('--camera', type=int, default=0) and change it to parser.add_argument('--camera', type=str, default=0) type string instead of int.

## Running the code:

11.	Now for the exciting part, running our code. Ensure that you have any video handy to test your pose estimation with. And then we type in the following

``` python run_webcam.py --camera=PATH_TO_VIDEO```  or ``` python run_webcam.py --camera=PATH_TO_VIDEO --model=cmu```

If you have the video in the same folder you can just type (for example):

``` python run_webcam.py --camera video.mp4``` 

12.	And for webcam ``` python run_webcam.py’ or ‘python run_webcam.py –model=cmu```

And there you have it! Quick and easy pose estimation in under 10 minutes. In the following lectures I will show you how to build some really exciting apps like people counting, fall detection and even yoga pose correction. Hope you enjoyed this tutorial.
Thank you for watching and we’ll see you in the next video. 





