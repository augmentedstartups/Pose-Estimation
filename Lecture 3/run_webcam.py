import argparse
import logging
import time
from pprint import pprint
import cv2
import numpy as np
import sys
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import os
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default=0)
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--save_video',type=bool,default=False, 
                        help='To write output video. default name file_name_output.avi')
    args = parser.parse_args()
    
    print("mode 0: Only Pose Estimation \nmode 1: People Counter \nmode 2: Fall Detection")
    mode = int(input("Enter a mode : "))
    
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    if(args.camera == '0'):
        file_write_name = 'camera_0'
    else:
        basename = os.path.basename(args.camera)
        # path = os.path.dirname(imgfile)
        file_write_name, _ = os.path.splitext(args.camera) 
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    count = 0
    y1 = [0,0]
    frame = 0
    while True:
        ret_val, image = cam.read()
        i =1
        count+=1
        if count % 11 == 0:
            continue
        # logger.debug('image process+')
        if not ret_val:
            break
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        # In humans total num of detected person in frame
        # logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        # logger.debug('show+')
        if mode == 1:
            hu = len(humans)
            print("Total no. of People : ", hu)
        elif mode == 2:
            for human in humans:
                # we select one person from num of person
                for i in range(len(humans)):
                    try:
                        '''
                        To detect fall we have used y coordinate of head.
                        Coordinates of head in form of normalize form.
                        We convert normalized points to relative point as per the image size.
                        y1.append(y) will store y coordinate to compare with previous point.
                        We have used try and except because many time pose estimator cann't predict head point.
                        
                        '''
                        #human.parts contains all the detected body parts
                        a = human.body_parts[0]   # human.body_parts[0] is for head point coordinates
                        x = a.x*image.shape[1]   # x coordinate relative to image 
                        y = a.y*image.shape[0]   # y coordinate relative to image
                        y1.append(y)   # store value of y coordinate in list to compare two frames
                    except:
                        pass
                    if ((y - y1[-2]) > 25):  # it's distance between frame and comparing it with thresold value 
                        cv2.putText(image, "Fall Detected", (20,50), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0,0,255), 
                            2, 11)
                        print("fall detected.",i+1, count)#You can set count for get that your detection is working
        elif mode == 0:	
        	pass
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if(frame == 0) and (args.save_video):   # It's use to intialize video writer ones
            out = cv2.VideoWriter(file_write_name+'_output.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                    20,(image.shape[1],image.shape[0]))
        out.write(image)
        if cv2.waitKey(1) == 27:
            break
        # logger.debug('finished+')

    cv2.destroyAllWindows()