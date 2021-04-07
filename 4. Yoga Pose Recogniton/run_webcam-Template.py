#=====================================================
#Modified by: Augmented Startups & Geeky Bee AI
#Date : 22 April 2019
#Project: Yoga Angle Corrector/Plank Calc/Body Ratio
#Tutorial: http://augmentedstartups.info/OpenPose-Course-S
#=====================================================
import argparse
import logging
import time
from pprint import pprint
import cv2
import numpy as np
import sys
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")



def find_point(pose, p):
    for point in pose:
            try:
                body_part = point.body_parts[p]
                return (int(body_part.x * width + 0.5), int(body_part.y * height + 0.5))
            except:
                return (0, 0)

    return (0, 0)


def euclidian(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def angle_calc(p0, p1, p2):
    '''
        p1 is center point from where we measured angle between p0 and
    '''

    try:
        a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
        b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
        angle = math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/math.pi
    except:
        return 0
    return int(angle)



def plank(a, b, c, d, e, f):
    #There are ranges of angle and distance to for plank. 
    '''
        a and b are angles of hands
        c and d are angle of legs
        e and f are distance between head to ankle because in plank distace will be maximum.
    '''
    if (a in range(50,100) or b in range(50,100)) and (c in range(135,175) or d in range(135,175)) and (e in range(50,250) or f in range(50,250)):
        return True
    return False



def mountain_pose(a, b, c, d, e):
    '''
        a is distance between two wrists
        b and c are angle between neck,shoulder and wrist 
        e and f are distance between head to ankle because in plank distace will be maximum.
    '''
    if a in range(20,160) and b in range(60,140) and c in range(60,140) and d in range(100,145) and e in range(100,145):
        return True
    return False




def draw_str(dst, xxx_todo_changeme, s, color, scale):
    (x, y) = xxx_todo_changeme
    if (color[0]+color[1]+color[2]==255*3):
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, (0, 0, 0), thickness = 4, lineType=10)
    else:
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, scale, color, thickness = 4, lineType=10)
    #cv2.line    
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, scale, (255, 255, 255), lineType=11)







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default="0")

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()



    print("mode 0: Only Pose Estimation \nmode 1: People Counter \nmode 2: Fall Detection \nmode 3: Yoga pose angle Corrector \nmode 4: Planking/Push up Detection \nmode 5: Hourglass ratio")
    mode = int(input("Enter a mode : "))

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    steam_name = args.camera
    if len(args.camera) == 1:
        steam_name = int(args.camera)
    # else\
    cam = cv2.VideoCapture(steam_name)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    count = 0
    i = 0
    frm = 0
    y1 = [0, 0]
    global height, width
    orange_color = (0, 140, 255)

    while True:
        ret_val, image = cam.read()

        i = 1

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        pose = humans
    
        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        height, width = image.shape[0], image.shape[1]

        if mode == 1:
            hu = len(humans)
            # print("Total no. of People : ", hu)

        elif mode == 2:
            for human in humans:
                for i in range(len(humans)):
                    try:
                        a = human.body_parts[0] #Head point
                        x = a.x*image.shape[1]
                        y = a.y*image.shape[0]
                        y1.append(y)
                    except:
                        pass
                    if ((y - y1[-2]) > 30):
                        print("fall detected.",i+1, count)#You can set count for get that your detection is working

        
        elif mode == 3:
            
            if len(pose) > 0:
                # distance calculations
                pass
                # angle calculations



        elif mode == 4:
            null
        elif mode == 5:
            null   



        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        #image = cv2.resize(image, (720, 720))
        if (frm == 0 ):
            out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (image.shape[1],image.shape[0]))
            print("Initializing")
            frm+=1
        cv2.imshow('tf-pose-estimation result', image)
        if i != 0:
            out.write(image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
