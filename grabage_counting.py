import tensorflow as tf

# Object detection imports
from utils import backbone
from api import object_counting_api

input_video = "./input_images_and_videos/demo (5).mp4"

detection_graph, category_index = backbone.set_model('custom_trained_inference_graph', 'label_map.pbtxt')

is_color_recognition_enabled = 1 
roi = 385 # roi line position
deviation = 1 # the constant that represents the object counting area

object_counting_api.cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, roi, deviation) 
