#! /usr/bin/env python
# -*- coding: utf-8 -*-

from app.main import bp
from flask import render_template, request
from werkzeug import secure_filename
import os
import json
import PIL
from PIL import Image
import numpy as np
from recognizer import predict_with_path


def print_image(image):
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            dot = image[i][j]
            if dot < 10:
                print("  ", end="")
            else:
                print("##", end="")

        print("")


def resize_image(image, size):
    img = PIL.Image.fromarray(np.uint8(image))
    img = img.resize((size, size), PIL.Image.ANTIALIAS)

    return img


@bp.route("/")
@bp.route("/index")
def index():
    return render_template("index.html")


@bp.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    result = {}

    if file:
        filename = secure_filename(file.name) + '.jpeg'
        file.save(os.path.join('.uploads', filename))
        result.update({"saved": True})
        result.update({"file": filename})
        pred = predict_with_path(os.path.join('.uploads', filename))
        result.update({"predict": int(pred[0])})

    else:
        result.update({"saved": False})

    return json.dumps(result)
