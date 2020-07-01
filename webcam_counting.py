import tensorflow as tf

from utils import backbone
from api import object_counting_api

 
detection_graph, category_index = backbone.set_model('custom_trained_inference_graph', 'label_map.pbtxt')

is_color_recognition_enabled = 0

object_counting_api.object_counting_webcam(detection_graph, category_index, is_color_recognition_enabled)
