import numpy as np
from PIL import Image
import image_processing
import os
from flask import Flask, render_template, request, make_response
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
import cv2
import random
from colorization.demo_release import colorize_and_save
import torch

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
    return update_wrapper(no_cache, view)


@app.route("/index")
@app.route("/")
@nocache
def index():
    return render_template("index.html", file_paths="img/image_here.jpg")


@app.route("/about")
@nocache
def about():
    return render_template('about.html')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/upload", methods=["POST"])
@nocache
def upload():
    target = os.path.join(APP_ROOT, "static/img")
    if not os.path.isdir(target):
        if os.name == 'nt':
            os.makedirs(target)
        else:
            os.mkdir(target)
    for file in request.files.getlist("file"):
        file.save("static/img/img_grey.jpg")
    copyfile("static/img/img_grey.jpg", "static/img/img_normal.jpg")
    
    return render_template("uploaded.html", file_paths=["img/img_normal.jpg"])

@app.route("/colorization", methods=["POST"])
@nocache
def colorization():
    print(torch.version.cuda)
    # Gantilah 'path_to_grayscale_image.jpg' dengan path gambar grayscale Anda
    img, img_bw, out_img_eccv16, out_img_siggraph17 = colorize_and_save('img/apple.jpg', 'output_prefix', use_gpu=True)

    # Simpan hasil ke direktori static/img/
    output_folder = 'static/img'
    img_path = f'{output_folder}/{output_prefix}_original.jpg'
    img_bw_path = f'{output_folder}/{output_prefix}_input.jpg'
    out_img_eccv16_path = f'{output_folder}/{output_prefix}_eccv16.jpg'
    out_img_siggraph17_path = f'{output_folder}/{output_prefix}_siggraph17.jpg'

    
    return render_template("colorization.html", file_paths=["img/img_normal.jpg"])

@app.route("/grayscale", methods=["POST"])
@nocache
def grayscale():
    image_processing.grayscale()
    #return render_template("uploaded.html", file_path="img/img_now.jpg")
    return histogram_rgb()


@app.route("/histogram_rgb", methods=["POST"])
@nocache
def histogram_rgb():
    img = Image.open("static/img/img_now.jpg")
    lebar, tinggi = (img).size
    image_processing.histogram_rgb()
    if image_processing.is_grey_scale("static/img/img_now.jpg"):
        return render_template("uploaded_2.html", file_paths=["img/img_normal.jpg","img/img_now.jpg","img/grey_histogram.jpg"], lebar=lebar, tinggi=tinggi)
    else:
        return render_template("uploaded_2.html", file_paths=["img/img_normal.jpg","img/img_now.jpg","img/red_histogram.jpg", "img/green_histogram.jpg", "img/blue_histogram.jpg"], lebar=lebar, tinggi=tinggi)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
