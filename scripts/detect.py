#!/usr/bin/env python3

import roslib
roslib.load_manifest("edge_tpu")
import sys
import rospy
import cv2
import time
from std_msgs.msg import String
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import PIL

import edgetpu.detection.engine

class tpu_detector:

    def __init__(self, model, labels, threshold=0.5, device_path=None):
        self.bridge = CvBridge()
        rospy.loginfo("Loading model {}".format(model))
        self.image_sub = rospy.Subscriber("input", Image, self.callback)
        self.threshold = threshold
        self.engine = edgetpu.detection.engine.DetectionEngine(model, device_path)
        rospy.loginfo("Engine loaded")
        self.load_labels(labels)
        self.detection_pub = rospy.Publisher('detections', Detection2DArray, queue_size=10)

    def load_labels(self, labels):
        with open(labels, 'r', encoding="utf-8") as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)

        results = self.engine.DetectWithImage(PIL.Image.fromarray(cv_image), top_k=1, threshold=self.threshold, keep_aspect_ratio=True, relative_coord=True)
        
        detections = Detection2DArray()
        now = rospy.get_rostime()

        for detection in results:
        
            top_left, bottom_right = detection.bounding_box
            min_x, min_y = top_left
            max_x, max_y = bottom_right
            
            imheight, imwidth, _ = cv_image.shape
            
            min_x *= imwidth
            max_x *= imwidth
            min_y *= imheight
            max_y *= imheight
            
            centre_x = (max_x+min_x)/2.0
            centre_y = (max_y+min_y)/2.0
            height = max_y-min_y
            width = max_x-min_x
            
            if height <=0 or width <= 0:
              continue
            
            bbox = BoundingBox2D()
            bbox.center.x = centre_x
            bbox.center.y = centre_y
            bbox.size_x = width
            bbox.size_y = height
            
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = detection.label_id
            hypothesis.score = detection.score
            hypothesis.pose.pose.position.x = centre_x
            hypothesis.pose.pose.position.y = centre_y
 
            # update the timestamp of the object
            object = Detection2D()
            object.header.stamp = now
            object.header.frame_id = data.header.frame_id
            object.results.append(hypothesis)
            object.bbox = bbox
            object.source_img.header.frame_id = data.header.frame_id
            object.source_img.header.stamp = now
            object.source_img.height = int(height)
            object.source_img.width = int(width)
            object.source_img.encoding = "rgb8"
            object.source_img.step = int(width*3)
            object.source_img.data = cv_image[int(min_y):int(max_y), int(min_x):int(max_x)].tobytes()
            
            detections.detections.append(object)
            
        if len(results) > 0:
          self.detection_pub.publish(detections)
        
        rospy.logdebug("%.2f ms" % self.engine.get_inference_time())

def main(args):

    rospy.init_node('detect', anonymous=True)
    
    model_path = rospy.get_param('~model_path')
    label_path = rospy.get_param('~label_path')
    threshold = rospy.get_param('~threshold', default=0.5)
    device_path = rospy.get_param('~device_path', default=None)

    detector = tpu_detector(model_path, label_path, threshold, device_path)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == "__main__":
    main(sys.argv)
