import pickle
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import numpy as np
import math
import random


def load_driving_log(directory):
    with open(directory + "/driving_log.pickle", "rb") as f:
        return pickle.load(f)

def preprocess_image(img):
    #crop image to remove hood and background
    # remainder is 50 x 320
    img = img.crop((0, 60, 320, 110))
    # resize to 32x32
    #img = img.resize((32, 32), resample=Image.BICUBIC)
    img = img.resize((200, 66), resample=Image.BICUBIC)
    img = np.array(img).reshape(66, 200,3)
    return np.array(img).astype('float32')
    #return np.array(img).astype('float32') / 255.0
    # think about YUV conversion

def rotate_image(img, steering_angle):
    angle = np.random.uniform(-5.0, 5.0)
    return img.rotate(angle, resample=Image.BICUBIC, expand = 0), steering_angle - angle

def mirror_image(img, steering_angle):
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
        steering_angle *= -1
    return img, steering_angle


def ImageGenerator(log_data, batch_size=128, augment_data=True):
    num_images = len(log_data)
    idx = 0
    side_correction = 5.0

    while True:
        X = []
        y = []
        for i in range(0,batch_size):
            file_name, steering_angle, throttle, brake, speed = log_data[idx]
            idx = (idx + 1) % num_images

            img = Image.open(file_name)
            dir_name, _, file_name = file_name.split('/')
            # adjust steering angle for left and right camera, respectively
            if "right" in file_name:
                steering_angle -= side_correction
            elif "left" in file_name:
                steering_angle += side_correction

            if (augment_data):
                img, steering_angle = rotate_image(img, steering_angle)
                img, steering_angle = mirror_image(img, steering_angle)

            # rescale and normalize
            img_data = preprocess_image(img)

            X.append(img_data)
            y.append(steering_angle)

        yield np.array(X), np.array(y)
