# This is a modification of John Chen's Agile trainer.
# I changed the code to adapt and improve keyboard input, so data with 
# higher quality can be recorded
#

## Import some useful modules
import argparse
import base64
import json
import cv2
import pygame
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.optimizers import Adam
from keras.layers.core import K
K.set_learning_phase(1)
import os
from os import path
from pathlib import Path
from threading import Thread
from time import sleep
from numpy.random import random
import pickle

from skimage.measure import compare_ssim as ssim


### initialize pygame and joystick
### modify for keyboard starting here!
img_rows, img_cols, ch = 66, 200, 3
pygame.init()
#pygame.joystick.init()
size = (img_cols, img_rows)
pygame.display.set_caption("Udacity SDC Project 3: camera video viewer")
screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)
sim_img = pygame.surface.Surface((img_cols,img_rows),0,24).convert()

#log_file = open("tmp_img/log_file.log", "a")

### PyGame screen diag output
# ***** get perspective transform for images *****
from skimage import transform as tf

rsrc = \
 [[43.45456230828867, 118.00743250075844],
  [104.5055617352614, 69.46865203761757],
  [114.86050156739812, 60.83953551083698],
  [129.74572757609468, 50.48459567870026],
  [132.98164627363735, 46.38576532847949],
  [301.0336906326895, 98.16046448916306],
  [238.25686790036065, 62.56535881619311],
  [227.2547443287154, 56.30924933427718],
  [209.13359962247614, 46.817221154818526],
  [203.9561297064078, 43.5813024572758]]
rdst = \
 [[10.822125594094452, 1.42189132706374],
  [21.177065426231174, 1.5297552836484982],
  [25.275895776451954, 1.42189132706374],
  [36.062291434927694, 1.6376192402332563],
  [40.376849698318004, 1.42189132706374],
  [11.900765159942026, -2.1376192402332563],
  [22.25570499207874, -2.1376192402332563],
  [26.785991168638553, -2.029755283648498],
  [37.033067044190524, -2.029755283648498],
  [41.67121717733509, -2.029755283648498]]

tform3_img = tf.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))

X = []
Y = []

def perspective_tform(x, y):
    p1, p2 = tform3_img((x,y))[0]
    return p2, p1

# ***** functions to draw lines *****
def draw_pt(img, x, y, color, shift_from_mid, sz=1):
    col, row = perspective_tform(x, y)
    row = int(row) + shift_from_mid
    col = int((col+img.get_height()*2)/3)
    if row >= 0 and row < img.get_width()-sz and\
       col >= 0 and col < img.get_height()-sz:
        img.set_at((row-sz,col-sz), color)
        img.set_at((row-sz,col), color)
        img.set_at((row-sz,col+sz), color)
        img.set_at((row,col-sz), color)
        img.set_at((row,col), color)
        img.set_at((row,col+sz), color)
        img.set_at((row+sz,col-sz), color)
        img.set_at((row+sz,col), color)
        img.set_at((row+sz,col+sz), color)

def draw_path(img, path_x, path_y, color, shift_from_mid):
    for x, y in zip(path_x, path_y):
        draw_pt(img, x, y, color, shift_from_mid)

# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
    deg_to_rad = np.pi/180.
    slip_fator = 0.0014 # slip factor obtained from real data
    steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
    wheel_base = 2.67   # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

    angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
    curvature = angle_steers_rad/(steer_ratio * wheel_base * (1. + slip_fator * v_ego**2))
    return curvature

def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
    #*** this function returns the lateral offset given the steering angle, speed and the lookahead distance
    curvature = calc_curvature(v_ego, angle_steers, angle_offset)

    # clip is to avoid arcsin NaNs due to too sharp turns
    y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999))/2.)
    return y_actual, curvature

def draw_path_on(img, speed_ms, angle_steers, color=(0,0,255), shift_from_mid=0):
    path_x = np.arange(0., 50.1, 0.5)
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
    draw_path(img, path_x, path_y, color, shift_from_mid)

### Preprocessing...
def preprocess(image):
    ## Preprocessing steps for your model done here:
    ## TODO:  Update if you need preprocessing!
    return image

### drive.py initialization
sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


DEFAULT_KEY_ANGLE = 0.183
DEFAULT_DECAY_TIME = 5
decay = 0
key_angle = 0
mixed_image = np.zeros((img_rows,img_cols,ch), dtype=np.float32)
import matplotlib.pyplot as plt

@sio.on('telemetry')
def telemetry(sid, data):
    global decay
    global key_angle
    global adam_lr

    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_prep = np.asarray(image, dtype=np.float32)
    image_array = cv2.resize(image_prep, (img_cols, img_rows), cv2.INTER_AREA).astype(np.float32)
    image_array /= 255
    image_array -= 0.5
    s = ssim(image_array, mixed_image, win_size=3)

    ### recording flag is replaced with training flag to start image data collection
    alpha = 0.5
    mixed_image[:,:,:] = cv2.addWeighted(image_array, alpha, mixed_image, 1-alpha, 0)
    transformed_image_array = mixed_image[None, :, :, :] 
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    predicted_angle = steering_angle

    if decay:
        key_angle *= 0.9
        steering_angle = key_angle
        decay -= 1

    def set_decay(decay):
        if decay == 0:
            decay = DEFAULT_DECAY_TIME
        else:
            decay += 1
        return decay

    #sharp_factor = 1 + decay / (1 * (decay + 1))
    sharp_factor = 1 + decay / 15
    keys = pygame.key.get_pressed() 
    if keys[pygame.K_LEFT]: 
        steering_angle = key_angle = -DEFAULT_KEY_ANGLE * sharp_factor
        decay = set_decay(decay)
    if keys[pygame.K_RIGHT]: 
        steering_angle = key_angle = DEFAULT_KEY_ANGLE * sharp_factor
        decay = set_decay(decay)
    if keys[pygame.K_DOWN]: 
        steering_angle *= 0.5
    if keys[pygame.K_UP]: 
        steering_angle *= 1.25
    if keys[pygame.K_0]: 
        steering_angle = key_angle = 0
        decay = 0

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            sharp_factor = 1 + decay / (1 * (decay + 1))
            if event.key == pygame.K_LEFT:
                steering_angle = key_angle = -DEFAULT_KEY_ANGLE * sharp_factor
                decay = set_decay(decay)
            elif event.key == pygame.K_RIGHT:
                steering_angle = key_angle = DEFAULT_KEY_ANGLE * sharp_factor
                decay = set_decay(decay)
            if event.key == pygame.K_s:
                save_model(model)
            elif event.key == pygame.K_b:
                clear_data()
            elif event.key == pygame.K_t:
                train_model(model)
            elif event.key == pygame.K_LEFTBRACKET:
                adam_lr = change_lr(model, 0.1, adam_lr)
            elif event.key == pygame.K_RIGHTBRACKET:
                adam_lr = change_lr(model, 10, adam_lr)
            elif event.key == pygame.K_q:
                plt.imshow(mixed_image)
                plt.show()


    # This model currently assumes that the features of the model are just the images. Feel free to change this.

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.3
    print("frame info:", round(steering_angle, 4), throttle, s, decay)
    send_control(steering_angle, throttle)

#    un-comment following code to enable data recording.
#    from time import time
#    tmp_fname = "img_%s.jpg" % str(round(time() * 1000))
#    tmp_fname = path.join(path.abspath('.'), 'tmp_img', tmp_fname)
#    save_image = cv2.cvtColor(image_prep, cv2.COLOR_BGR2RGB)
#    cv2.imwrite(tmp_fname, save_image)
#    log_file.write("{fn}\t{st:.4f}\t{pst:.4f}\t{d}\t{t:.2f}\n"\
#         .format(fn=tmp_fname, st=steering_angle, pst=predicted_angle, d=decay, t=throttle))
#    log_file.flush()


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)

def batchgen(X, Y):
    while 1:
        i = int(random()*len(X))
        y = Y[i]
        image = X[i]
        y = np.array([[y]])
        image = image.reshape(1, img_cols, img_rows, ch)
        yield image, y


def save_model(model):
    fileWeights = weights_file
    print("Saving model weights to disk: ", fileWeights)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)
    print("Model weights saved.")

def clear_data():
    print("X, Y cleared")
    X.clear()
    Y.clear()

def train_model(model):
    print("Training starts")
    batch_size = len(X)
    if batch_size == 0:
        print("Data buffer is empty")
        return
    nb_epoch = 10
    history = model.fit(np.stack(X), np.array(Y), batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

def change_lr(model, delta, adam_lr):
    model.optimizer.lr *= delta
    adam_lr *= delta
    print("change learning rate by", delta, "to", adam_lr)
    return adam_lr

def model_trainer(fileModelJSON, model):
    print("Model Trainer Thread Starting...")

    fileWeights = fileModelJSON.replace('json', 'h5')

    # start training loop...
    while 1:
        #if True:
        if len(X) > 50:
            batch_size = 20
            samples_per_epoch = int(len(X)/batch_size)
            val_size = int(samples_per_epoch/10)
            if val_size < 10:
                val_size = 10
            nb_epoch = 50

#            history = model.fit_generator(batchgen(X,Y),
#                                samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
#                                validation_data=batchgen(X,Y),
#                                nb_val_samples=val_size,
#                                verbose=1)
            
            history = model.fit(np.stack(X), np.array(Y), batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
            X.clear()
            Y.clear()
        else:
            print("Not Ready!  Sleeping for 5...")
            sleep(5)

def reporter():
    print("Data length: %s. (t)raining, (b)clear" % len(X))
    sleep(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()


    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)
    adam_lr = 0.00001
    adam = Adam(lr=adam_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])


    # start training thread
    thread = Thread(target = reporter)
    thread.start()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

    # wait for training to end
    thread.join()

