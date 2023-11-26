import numpy as np
from PIL import Image
import image_processing
import os
from flask import Flask, render_template, request, make_response,  url_for, redirect, flash
from datetime import datetime
from functools import wraps, update_wrapper
from shutil import copyfile
import cv2
import random
from colorization.demo_release import colorize_and_save
import torch
from sklearn.cluster import MiniBatchKMeans
from matplotlib.colors import rgb2hex
from collections import Counter


app = Flask(__name__)
app.config['SECRET_KEY'] = 'klasifikasi-warna'
app.config['UPLOAD_PATH'] = "static"
app.config['SESSION_TYPE'] = 'filesystem'

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

def to_dictionary(keys, values):
    return dict(zip(keys, values))

def process_image(image_path):
    # Buka citra yang akan diklasifiasi
    img_object = Image.open(fp=image_path, mode="r")

    # Simpan citra dengan nama file yang sudah ditentukan
    custom_filename = "color.png"  # Replace with your desired filename
    img_object.save(os.path.join(app.config['UPLOAD_PATH'], custom_filename))

    # Konversi citra ke 2d array
    img_array = np.array(img_object)
    
    # Gunakan MiniBatchKMeans algoritma dengan mengatur cluster n sebanyak 7
    k_warna = MiniBatchKMeans(n_clusters=7)
    k_warna.fit(img_array.reshape(-1, 3))

    # Hitung berapa banyak pixel per cluster
    n_pixels = len(k_warna.labels_)
    counter = Counter(k_warna.labels_)

    # 2D array yang berisi nilai pixel RGB sejumlah n cluster = 7
    rgb_int = k_warna.cluster_centers_

    # Konversi ke float 0-1
    rgb_float = np.array(rgb_int / 255, dtype=float)

    # Ensure RGB values are within the correct range
    rgb_float = np.clip(rgb_float, 0, 1)

    # Konversi nilai RGB ke nilai HEXA dan simpan ke dalam list
    hex_values = [rgb2hex(rgb_float[i, :]) for i in range(rgb_float.shape[0])]

    prop_warna = {}

    # Kalkulasi presentase tiap pixel warna
    for i in counter:
        prop_warna[i] = np.round(counter[i] / n_pixels, 4)

    # Konversi dictionary ke dalam list
    prop_warna = dict(sorted(prop_warna.items()))
    props_list = [value for (key, value) in prop_warna.items()]

    # Merge 2 list ke dalam sebuah dictionary
    dict_warna = to_dictionary(props_list, hex_values)

    # Sort/urutkan dict secara descending
    sorted_dict = dict(sorted(dict_warna.items(), reverse=True))

    return sorted_dict


@app.route("/colorization", methods=["POST"])
@nocache
def colorization():
    print(torch.version.cuda)
    # Gantilah 'path_to_grayscale_image.jpg' dengan path gambar grayscale Anda
    img, img_bw, out_img_eccv16, out_img_siggraph17 = colorize_and_save('static/img/img_normal.jpg', 'output_prefix', use_gpu=None)
            
   # Replace these paths with your grayscale image and output prefix
    grayscale_image_path = 'static/img/img_normal.jpg'
    output_prefix = 'output_prefix'
    
    img, img_bw, out_img_eccv16, out_img_siggraph17 = colorize_and_save(grayscale_image_path, output_prefix, use_gpu=None)

    sorted_dict1 = process_image("static/img/eccv16.png")
    sorted_dict2 = process_image("static/img/siggraph17.png")

    return render_template("colorization.html", file_paths=["img/img_normal.jpg", "img/eccv16.png", "img/siggraph17.png"], colors1=sorted_dict1, colors2=sorted_dict2)

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
