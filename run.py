import matplotlib
import matplotlib.pyplot as plt

import os
import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


def load_image_into_numpy_array(path):

  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

pipeline_file = 'research\deploy\pipeline_file.config'
model_dir = 'training'
pipeline_config = pipeline_file
#generally you want to put the last ckpt from training in here
model_dir = 'training\ckpt-1'
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
      model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(os.path.join('training\ckpt-3'))


def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

detect_fn = get_model_detection_function(detection_model)

label_map_path = configs['eval_input_config'].label_map_path
label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)










#change tabel path here
image_path = 'tab32.jpg'
image_np = load_image_into_numpy_array(image_path)

input_tensor = tf.convert_to_tensor(
    np.expand_dims(image_np, 0), dtype=tf.float32)
detections, predictions_dict, shapes = detect_fn(input_tensor)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'][0].numpy(),
      (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
      detections['detection_scores'][0].numpy(),
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.5,
      agnostic_mode=False,
)

img_height, img_width, img_channel = image_np.shape
absolute_coord = []
THRESHOLD = 0.7 # adjust your threshold here
N = len(detections['detection_boxes'][0].numpy())
for i in range(N):
    if detections['detection_scores'][0][i].numpy() < THRESHOLD:
        continue
    box = detections['detection_boxes'][0][i].numpy()
    ymin, xmin, ymax, xmax = box
    x_up = int(xmin*img_width)
    y_up = int(ymin*img_height)
    x_down = int(xmax*img_width)
    y_down = int(ymax*img_height)
    absolute_coord.append((x_up,y_up,x_down,y_down))


bounding_box_img = []
for c in absolute_coord:
    bounding_box_img.append(image_np[c[1]:c[3], c[0]:c[2],:])

print(len(bounding_box_img))
print(absolute_coord)


from keras.preprocessing.image import save_img
save_img('a.jpg', bounding_box_img[0])