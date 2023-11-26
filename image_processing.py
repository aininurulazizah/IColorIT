import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from collections import Counter
from pylab import savefig
import cv2 as cv
from flask import send_file
import os
import random


def crop(img_path, rows, cols):
    im = Image.open(img_path)
    w, h = im.size
    unit_width = w // cols
    unit_height = h // rows
    img_path_save = "static/img/puzzle"

    for row in range(rows):
        for col in range(cols):
            left = col * unit_width
            upper = row * unit_height
            right = left + unit_width
            lower = upper + unit_height
            im1 = im.crop((left, upper, right, lower))
            im1.save(os.path.join(img_path_save, f"crop_{row}_{col}.png"))
            
def crop_and_shuffle(img_path, rows, cols):
    im = Image.open(img_path)
    w, h = im.size
    unit_width = w // cols
    unit_height = h // rows
    img_path_save = "static/img/puzzle"

    count = 1  # Variabel untuk menghitung nama file
    for row in range(rows):
        for col in range(cols):
            left = col * unit_width
            upper = row * unit_height
            right = left + unit_width
            lower = upper + unit_height
            im1 = im.crop((left, upper, right, lower))
            
            # Simpan gambar dengan nama crop1, crop2, crop3, dst.
            im1.save(os.path.join(img_path_save, f"crop_{count}.png"))
            count += 1  # Tingkatkan variabel count


def grayscale():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]
    new_arr = r.astype(int) + g.astype(int) + b.astype(int)
    new_arr = (new_arr/3).astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    for i in range(w):
        for j in range(h):
            r, g, b = im.getpixel((i, j))
            if r != g != b:
                return False
    return True


def zoomin():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img)
    new_size = ((img_arr.shape[0] * 2),
                (img_arr.shape[1] * 2), img_arr.shape[2])
    new_arr = np.full(new_size, 255)
    new_arr.setflags(write=1)

    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]

    new_r = []
    new_g = []
    new_b = []

    for row in range(len(r)):
        temp_r = []
        temp_g = []
        temp_b = []
        for i in r[row]:
            temp_r.extend([i, i])
        for j in g[row]:
            temp_g.extend([j, j])
        for k in b[row]:
            temp_b.extend([k, k])
        for _ in (0, 1):
            new_r.append(temp_r)
            new_g.append(temp_g)
            new_b.append(temp_b)

    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            new_arr[i, j, 0] = new_r[i][j]
            new_arr[i, j, 1] = new_g[i][j]
            new_arr[i, j, 2] = new_b[i][j]

    new_arr = new_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def zoomout():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    x, y = img.size
    new_arr = Image.new("RGB", (int(x / 2), int(y / 2)))
    r = [0, 0, 0, 0]
    g = [0, 0, 0, 0]
    b = [0, 0, 0, 0]

    for i in range(0, int(x/2)):
        for j in range(0, int(y/2)):
            r[0], g[0], b[0] = img.getpixel((2 * i, 2 * j))
            r[1], g[1], b[1] = img.getpixel((2 * i + 1, 2 * j))
            r[2], g[2], b[2] = img.getpixel((2 * i, 2 * j + 1))
            r[3], g[3], b[3] = img.getpixel((2 * i + 1, 2 * j + 1))
            new_arr.putpixel((int(i), int(j)), (int((r[0] + r[1] + r[2] + r[3]) / 4), int(
                (g[0] + g[1] + g[2] + g[3]) / 4), int((b[0] + b[1] + b[2] + b[3]) / 4)))
    new_arr = np.uint8(new_arr)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_left():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if(is_grey_scale(img_path)):
        img_arr = np.roll(img_arr, shift=-50, axis=1)
    else:
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.roll(r, shift=-50, axis=1)
        g = np.roll(g, shift=-50, axis=1)
        b = np.roll(b, shift=-50, axis=1)
        img_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(img_arr)
    new_img.save("static/img/img_now.jpg")


def move_right():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if(is_grey_scale(img_path)):
        img_arr = np.roll(img_arr, shift=50, axis=1)
    else:
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.roll(r, shift=50, axis=1)
        g = np.roll(g, shift=50, axis=1)
        b = np.roll(b, shift=50, axis=1)
        img_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(img_arr)
    new_img.save("static/img/img_now.jpg")

def move_up():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if(is_grey_scale(img_path)):
        img_arr = np.roll(img_arr, shift=-50, axis=0)
    else:
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.roll(r, shift=-50, axis=0)
        g = np.roll(g, shift=-50, axis=0)
        b = np.roll(b, shift=-50, axis=0)
        img_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(img_arr)
    new_img.save("static/img/img_now.jpg")


def move_down():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if(is_grey_scale(img_path)):
        img_arr = np.roll(img_arr, shift=50, axis=0)
    else:
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        r = np.roll(r, shift=50, axis=0)
        g = np.roll(g, shift=50, axis=0)
        b = np.roll(b, shift=50, axis=0)
        img_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(img_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_addition():
    img = Image.open("static/img/img_now.jpg")
    if(img.mode == 'RGBA'):
        img = img.convert('RGB')
    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr+100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_substraction():
    img = Image.open("static/img/img_now.jpg")
    if(img.mode == 'RGBA'):
        img = img.convert('RGB')
    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr-100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_multiplication():
    img = Image.open("static/img/img_now.jpg")
    if(img.mode == 'RGBA'):
        img = img.convert('RGB')
    img_arr = np.asarray(img)
    img_arr = img_arr*1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_division():
    img = Image.open("static/img/img_now.jpg")
    if(img.mode == 'RGBA'):
        img = img.convert('RGB')
    img_arr = np.asarray(img)
    img_arr = img_arr/1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def convolution(img, kernel):
    h_img, w_img, _ = img.shape
    out = np.zeros((h_img-2, w_img-2), dtype=float)
    new_img = np.zeros((h_img-2, w_img-2, 3))
    if np.array_equal((img[:, :, 1], img[:, :, 0]), img[:, :, 2]) == True:
        array = img[:, :, 0]
        for h in range(h_img-2):
            for w in range(w_img-2):
                S = np.multiply(array[h:h+3, w:w+3], kernel)
                out[h, w] = np.sum(S)
        out_ = np.clip(out, 0, 255)
        for channel in range(3):
            new_img[:, :, channel] = out_
    else:
        for channel in range(3):
            array = img[:, :, channel]
            for h in range(h_img-2):
                for w in range(w_img-2):
                    S = np.multiply(array[h:h+3, w:w+3], kernel)
                    out[h, w] = np.sum(S)
            out_ = np.clip(out, 0, 255)
            new_img[:, :, channel] = out_
    new_img = np.uint8(new_img)
    return new_img


def edge_detection():
    zoomin()
    zoomout()
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=int)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")

def identity_kernel():
    zoomin()
    zoomout()
    img = cv.imread("static/img/img_now.jpg")
    
    kernel = np.array([[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]])
    identity = cv.filter2D(src = img, ddepth = -1, kernel = kernel)  
    cv.imwrite("static/img/img_now.jpg", identity) 

def blur():
    zoomin()
    zoomout()
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=int)
    kernel = np.array(
        [[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")
    
def sharpening():
    zoomin()
    zoomout()
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img, dtype=int)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    new_arr = convolution(img_arr, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")
    
def lowpass_blur(kernel_size):
    zoomin()
    zoomout()
    img = cv.imread("static/img/img_now.jpg")
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    blur = cv.filter2D(src=img, ddepth=-1, kernel=kernel)
    cv.imwrite("static/img/img_now.jpg", blur) 
    
def highpass(kernel_size):
    zoomin()
    zoomout()
    img = cv.imread("static/img/img_now.jpg")
    kernel = highpass_custom_kernel(kernel_size)
    highpass = cv.filter2D(img,-1,kernel)
    cv.imwrite("static/img/img_now.jpg", highpass) 
    
def bandpass(kernel_size):
    zoomin()
    zoomout()
    img = cv.imread("static/img/img_now.jpg")
    kernel = bandpass_custom_kernel(kernel_size)
    bandpass = cv.filter2D(img,-1,kernel)
    cv.imwrite("static/img/img_now.jpg", bandpass) 
    
def gaussian_blur(kernel_size):
    zoomin()
    zoomout()
    img = cv.imread("static/img/img_now.jpg")
    gaussian_blur = cv.GaussianBlur(src=img,ksize=(kernel_size,kernel_size),sigmaX=0)
    cv.imwrite("static/img/img_now.jpg", gaussian_blur) 
    
def median_blur(kernel_size):
    zoomin()
    zoomout()
    img = cv.imread("static/img/img_now.jpg")
    gaussian_blur = cv.medianBlur(src=img, ksize=kernel_size)
    cv.imwrite("static/img/img_now.jpg", gaussian_blur)
     
def bilateral(kernel_size):
    zoomin()
    zoomout()
    img = cv.imread("static/img/img_now.jpg")
    bilateral = cv.bilateralFilter(src = img, d = kernel_size, sigmaColor = 75, sigmaSpace = 75)
    cv.imwrite("static/img/img_now.jpg", bilateral) 
    
def zero_padding(size):
    zoomin()
    zoomout()
    img = cv.imread("static/img/img_now.jpg")
    padding = cv.copyMakeBorder(img, size, size, size, size, cv.BORDER_CONSTANT, value=0)
    cv.imwrite("static/img/img_now.jpg", padding) 
    

def histogram_rgb():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        g = img_arr.flatten()
        data_g = Counter(g)
        plt.bar(list(data_g.keys()), data_g.values(), color='black')
        plt.savefig(f'static/img/grey_histogram.jpg', dpi=300)
        plt.clf()
    else:
        r = img_arr[:, :, 0].flatten()
        g = img_arr[:, :, 1].flatten()
        b = img_arr[:, :, 2].flatten()
        data_r = Counter(r)
        data_g = Counter(g)
        data_b = Counter(b)
        data_rgb = [data_r, data_g, data_b]
        warna = ['red', 'green', 'blue']
        data_hist = list(zip(warna, data_rgb))
        for data in data_hist:
            plt.bar(list(data[1].keys()), data[1].values(), color=f'{data[0]}')
            plt.savefig(f'static/img/{data[0]}_histogram.jpg', dpi=300)
            plt.clf()


def df(img):  # to make a histogram (count distribution frequency)
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i, j]] += 1
    return values


def cdf(hist):  # cumulative distribution frequency
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1]+hist[i]
    # Now we normalize the histogram
    # What your function h was doing before
    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf


def histogram_equalizer():
    img = cv.imread('static\img\img_now.jpg', 0)
    my_cdf = cdf(df(img))
    # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
    image_equalized = np.interp(img, range(0, 256), my_cdf)
    cv.imwrite('static/img/img_now.jpg', image_equalized)


def threshold(lower_thres, upper_thres):
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    condition = np.logical_and(np.greater_equal(img_arr, lower_thres),
                               np.less_equal(img_arr, upper_thres))
    print(lower_thres, upper_thres)
    img_arr = img_arr.copy()
    img_arr[condition] = 255
    new_img = Image.fromarray(img_arr)
    new_img = new_img.convert('RGB')
    new_img.save("static/img/img_now.jpg")
    
def highpass_custom_kernel(size):
    # Hitung nilai tengah kernel
    center_value = (size * size) - 1

    custom_kernel = np.full((size, size), -1, dtype=np.float32)
    custom_kernel[size // 2, size // 2] = center_value

    return custom_kernel

def bandpass_custom_kernel(size):
    # Hitung nilai tengah kernel
    center_value = (size * size)

    custom_kernel = np.full((size, size), -1, dtype=np.float32)
    custom_kernel[size // 2, size // 2] = center_value

    return custom_kernel
