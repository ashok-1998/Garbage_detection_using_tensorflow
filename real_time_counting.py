import tensorflow as tf

# Object detection imports
from utils import backbone
from api import object_counting_api

input_video = "./input_images_and_videos/demo (6).mp4"

detection_graph, category_index = backbone.set_model('custom_trained_inference_graph', 'label_map.pbtxt')

is_color_recognition_enabled = 0

#object_counting_api.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting

object_counting_api.object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled) # counting all the objects
