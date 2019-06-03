import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import threading
from threading import Thread
import time
import requests

import label_map_util
import visualization_utils as vis_util

# Path to frozen detection graph. This are the actual models that is used for the object detection:
# Modelo para detectar bastones
PATH_TO_CKPT = 'frozen_inference_graph.pb'
#Modelo para detectar otros objetos
PATH_TO_CKPT2 =  'frozen_inference_graph_normal.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'object-detection.pbtxt')
PATH_TO_LABELS2 = os.path.join('data', 'mscoco_label_map.pbtxt')

#*********************CREACIÓN GRÁFICO DEL PRIMER MODELO**********************************************************************************************************
NUM_CLASSES = 1

FILENAME = 'video_test_hubdate1.mp4'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#****************************MODELO BASTONES**********************************************************************************
stop=0
def run_inference_for_single_image(graph, filename):
  with graph.as_default():
    with tf.Session() as sess:
      cap = cv2.VideoCapture(filename)
      y=0
      global stop
      while(True):
          global stop
          ret, image_np = cap.read()
          image_np_expanded = np.expand_dims(image_np, axis=0)
          image= image_np
          # Get handles to input and output tensors
          ops = tf.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          tensor_dict = {}
          for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
          if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          # Run inference
          output_dict = sess.run(tensor_dict,
                                feed_dict={image_tensor: np.expand_dims(image, 0)})
          #Convierto de numpy.float a float la probabilidad de deteción:
          x=np.float(output_dict['detection_scores'][0][0])
          #print(type(x))
          #print(x)
          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          #print(output_dict['detection_scores'])
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]

          # Visualization of the results of a detection.
          # Detectar bastones solo cuando es superior a cierta probabilidad:
          if x>0.8555565:
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)
            y=y+1
          # Mostrar detecciones:
          cv2.imshow("object detecion",cv2.resize(image_np,(600,400)))
          #cv2.imwrite('b.mp4', image_np)
          ###
          ###
          # Varias restrinciones en formas de if para solo activar la alerta cuando detecta un baston durante varios frames y
          # a partir de un tiempo desde el lanzamiento de la anterior alerta:
          
          if y>=40:
            if stop==0:
                stop=alerta()
                threads=list()
                t=threading.Thread(target=stopp)
                threads.append(t)
                t.start()
                y=0
          if cv2.waitKey(25) % 0xFF ==ord("q"):
           cv2.destroyAllWindows()
           break
      

      cap.release()
      cv2.destroyAllWindows()

#Función utilizada para evitar lanzar varias alertas muy seguidas
def stopp():
    global stop
    #Indicar el tiempo mínimo que habrá entre las alertas:
    time.sleep(20)
    print("Pasados 20 segundos ponemos stop en 0 otra vez")
    stop=0

#Funcion donde incluiremos los post hace JANE o hacia el entorno del gestor
def alerta():
    print("Alerta activada")
    print("Incluir la lógica de la alerta al detectar un invidente")
    threads = list()
    t = threading.Thread(target=post)
    threads.append(t)
    t.start()
    return 1

def post():
    r = requests.post("http://....")

# Main desde donde se puede elegir que objetos detectar:
def main():
   #Solo detectamos bastones:
   run_inference_for_single_image(detection_graph, FILENAME)
   #Detectamos personas y otros objetos:
   #run_inference_for_single_image2(detection_graph2)
def return_image(image):
  cv2.imwrite('t.jpg', image)
  yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')
main()
