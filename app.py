from __future__ import division, print_function
# coding=utf-8

import os
import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

from openpyxl.utils.dataframe import dataframe_to_rows

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import *
import csv
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, flash, send_from_directory

# Define a flask app
app = Flask(__name__)

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'uploads/'



# # Object Detection Part
# def load_image_into_numpy_array(path):
#
#   img_data = tf.io.gfile.GFile(path, 'rb').read()
#   image = Image.open(BytesIO(img_data))
#   (im_width, im_height) = image.size
#   return np.array(image.getdata()).reshape(
#       (im_height, im_width, 3)).astype(np.uint8)
#
# pipeline_file = 'research\deploy\pipeline_file.config'
# model_dir = 'training'
# pipeline_config = pipeline_file
# #generally you want to put the last ckpt from training in here
# model_dir = 'training\ckpt-1'
# configs = config_util.get_configs_from_pipeline_file(pipeline_config)
# model_config = configs['model']
# detection_model = model_builder.build(
#       model_config=model_config, is_training=False)
#
# # Restore checkpoint
# ckpt = tf.compat.v2.train.Checkpoint(
#       model=detection_model)
# ckpt.restore(os.path.join('training\ckpt-3'))
#
# def get_model_detection_function(model):
#   """Get a tf.function for detection."""
#
#   @tf.function
#   def detect_fn(image):
#     """Detect objects in image."""
#
#     image, shapes = model.preprocess(image)
#     prediction_dict = model.predict(image, shapes)
#     detections = model.postprocess(prediction_dict, shapes)
#
#     return detections, prediction_dict, tf.reshape(shapes, [-1])
#
#   return detect_fn
#
# detect_fn = get_model_detection_function(detection_model)
#
# label_map_path = configs['eval_input_config'].label_map_path
# label_map = label_map_util.load_labelmap(label_map_path)
# categories = label_map_util.convert_label_map_to_categories(
#     label_map,
#     max_num_classes=label_map_util.get_max_label_map_index(label_map),
#     use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
# label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
#
#





def model_predict(img_path):
    img = cv2.imread(img_path, 0)
    print(img.shape)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    img_bin = 255 - img_bin

    kernel_len = np.array(img).shape[1] // 100
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    cv2.imwrite(r"vertical.jpg", vertical_lines)

    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    cv2.imwrite(r"horizontal.jpg", horizontal_lines)

    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

    # Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(r"img_vh.jpg", img_vh)
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)

    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def sort_contours(cnts, method="left-to-right"):

        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)
    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 1000 and h < 500):
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])

    row = []
    column = []
    j = 0
    # Sorting the boxes to their respective row and column
    for i in range(len(box)):
        if (i == 0):
            column.append(box[i])
            previous = box[i]
        else:
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]
                if (i == len(box) - 1):
                    row.append(column)
            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
    center = np.array(center)
    center.sort()

    finalboxes = []
    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

    outer = []
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner = ''
            if (len(finalboxes[i][j]) == 0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                 finalboxes[i][j][k][3]
                    finalimg = bitnot[x:x + h, y:y + w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel, iterations=1)
                    erosion = cv2.erode(dilation, kernel, iterations=1)

                    out = pytesseract.image_to_string(erosion)
                    if (len(out) == 0):
                        out = pytesseract.image_to_string(erosion, config='--psm 3')
                    inner = inner + " " + out
                outer.append(inner)

    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
    print(dataframe)
    for i in dataframe.columns:
        dataframe[i] = dataframe[i].str.replace('\x0c', '')
        dataframe[i] = dataframe[i].str.replace('\n', '')
    dataframe = dataframe.applymap(lambda x: x.encode('unicode_escape').
                                   decode('utf-8') if isinstance(x, str) else x)

    print(dataframe)
    writer = pd.ExcelWriter('test.xlsx')
    dataframe.to_excel(writer)
    writer.save()
    print("DOne")



@app.route('/')
def base():
    return render_template("home.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('model.html')
    else:

        target = os.path.join(APP_ROOT, 'uploads/')
        print(target)

        if not os.path.isdir(target):
            os.mkdir(target)

        file = request.files['file']

        filename = file.filename
        destination = "/".join([target, filename])

        file.save(destination)

        model_predict(destination)
        # preds = model_predict(destination)
        # print(preds)
        # print(type(model))
        # print("before done")
        return render_template('output.html', image_file_name=filename)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)



@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/model")
def model():
    return render_template("model.html")


if __name__ == '__main__':
    app.run(debug=True)
