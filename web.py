import tensorflow as tf
import csv
import os
import cv2
import numpy as np
import glob, os, tarfile, urllib
from utils import label_map_util
from utils import visualization_utils as vis_util
from twilio.rest import Client

is_color_recognition_enabled = 0
# What model to download.
model_name = 'custom_trained_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
path_to_frozen_graph = model_name + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
path_to_labels = 'data/label_map.pbtxt'

num_classes = 4

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(path_to_labels)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


import cv2
#cap=cv2.VideoCapture(0) # 0 stands for very first webcam attach
url = 'http://192.168.2.4:8080/video'
cap=cv2.VideoCapture(url)
#filename='F:\sample.avi'
#codec=cv2.VideoWriter_fourcc('m','p','4','v')#fourcc stands for four character code
#framerate=30
#resolution=(640,480)
  
#VideoFileOutput=cv2.VideoWriter(filename,codec,framerate, resolution)
   
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        ret=True
        while (ret):
            ret, image_np=cap.read() 

        # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
           # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
  
              # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
              # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
              # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                  #cap.get(1),
                  image_np,
                  #1,
                  #is_color_recognition_enabled,
                  np.squeeze(boxes),
                  np.squeeze(classes).astype(np.int32),
                  np.squeeze(scores),
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=8)
 
            #VideoFileOutput.write(image_np)
            cv2.imshow('live_detection',image_np)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        cv2.destroyAllWindows()
        cap.release()
