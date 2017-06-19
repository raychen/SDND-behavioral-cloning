# The code for original training process is in the notebook
# here I extracted code that does data loading, augmentation
# along with the model architect 

# To see the architect, check the method #model_init


import keras
import pandas as pd
import numpy as np

from os import path
from os.path import getmtime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from keras.models import model_from_json

import cv2
from tqdm import tqdm
from scipy import signal


from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from time import time


# Hyper parameters
# clip the steering angle, the value of steering is suppressed to [-MAX_STEERING, MAX_STEERING]
MAX_STEERING = 0.17

# for mixed_butter_filter, the idea is to retain certain level of sharp turn.
HIGH_FREQ_RATIO = 0.5

# adjust value used for left and right cameras
CAMERA_SHIFT = 0.08

# in the data augmentation for the 2nd phase training
# the amount of steering angle adjusted for each pixel shifted
ANGLE_PER_PIXEL = 0.05 / 40

# model file name
MODEL_NAME = 'model_%s' % int(time())


def butter_lowpass(x,fcut,f_sample,order,plen): 
    # fcut : cutoff frequency
    # f_sample : sampling frequency
    # order : Order of filter (4 typically)
    # plen: padding length (0 typically)
    
    rat = fcut/f_sample

    b, a = signal.butter(order, rat)
    y = signal.filtfilt(b, a, x, padlen=plen)
    return y


fcut = 40
def mixed_butter_filter(s):
    """
    clip and smooth the steering value
    use two butter_lowpass filters to smooth the signal
    """
    clipped_steer = np.clip(s, -max_steering, max_steering)
    component1 = butter_lowpass(clipped_steer, fcut, 500, 4, 0) * HIGH_FREQ_RATIO 
    component2 = butter_lowpass(clipped_steer, fcut, 1600, 4, 0) * (1 - HIGH_FREQ_RATIO)
    return component1 + component2


# from lane_finding import lane_finding

def read_image(fname):
#     _, basename = path.split(fname)
#     full_fname = path.join(prefix, basename)
    img = cv2.imread(fname.strip())
    return img


def pre_processing(img, resize_to):
    #resize
    img = cv2.resize(img, resize_to, cv2.INTER_AREA)

    #normalization
    img = img / 255 - 0.5
    return img



# DataGenerator for first phase training.
# Admittedly, some features are a little bit overkill.
class DataGenerator(object):
    
    def __init__(self, log_file, stack_x=None, skip_stack=True,
                 batch_size=32, resize_to=(200, 66),
                 steering_filter=None,
                 cameras="center left right".split(),
                 shift_value=(0, CAMERA_SHIFT, -CAMERA_SHIFT)):

        self.resize_to = resize_to
        self.log_file = log_file
        self.batch_size = batch_size
        self.camera = cameras
        self.shift_value = shift_value
        self.idx = 0
        self.log_file['mtime'] = self.log_file['center'].map(lambda f: getmtime(f))
        self.steering = self.preprocess_y(steering_filter)
        self.stack_x = stack_x
        if stack_x and not skip_stack:
            self.preprocess_x(stack_x, nb_stack=10, diff_tolerance=0.5)
        self.dataset_length = len(self.log_file)
        

    
    def time_aware_datagroups(self, mtime, diff_tolerance):
        """
        generate data points that is considered continous
        e.g. time difference within the data no lager than diff_tolerance)
        """
        gaps = mtime[mtime.diff(periods=1) > diff_tolerance].index
        
        # no time gap lager than tolerance
        if len(gaps) == 0:
            yield 0, len(mtime)

        for idx in tqdm(range(len(gaps))):
            begin = 0 if idx == 0 else gaps[idx-1]
            end = gaps[idx]
            yield begin, end
        
        yield gaps[idx], len(mtime)
        

    def preprocess_x(self, new_folder, nb_stack=5, diff_tolerance=1, alpha=0.5):
        """
        use cv2.addWeighted to 'stack' pictures,
        so each picture is a weighted sum of last #nb_stack of pictures
        the weights are decreased with a factor of 0.5 for each consecutive picture.
        This method is an attempt to reduce the sharp bend caused by keyboard control.
        
        The stacked pictures are stored in a sub-directory in original data directory
        """
        for begin, end in self.time_aware_datagroups(self.log_file.mtime, diff_tolerance):
            if end - begin < nb_stack - 1:
                continue
            for idx in self.log_file.index[begin:end]:
                lag = max(idx - nb_stack + 1, 0)
                for c in self.camera:
                    images = self.log_file.ix[lag:idx][c]\
                            .map(lambda fname: read_image(fname))
                    images = np.stack(images)
                    base = images[0]
                    for i in range(len(images)):
                        base = cv2.addWeighted(images[i], alpha, base, 1-alpha, 0)
                    new_fname = self.log_file[c][idx].replace("IMG", new_folder)
                    cv2.imwrite(new_fname.strip(), base)


    def preprocess_y(self, steering_filter, diff_tolerance=3):
        if not steering_filter:
            return self.log_file['steer']

        filtered_steer = np.zeros(len(self.log_file))
        for begin, end in self.time_aware_datagroups(self.log_file.mtime, diff_tolerance):
            filtered_steer[begin:end] = steering_filter(self.log_file[begin:end].steer)
        return filtered_steer
            
    def generate_X(self, sample_indices, ch=3):
        data_shape = (self.batch_size * len(self.camera),
                      self.resize_to[1],
                      self.resize_to[0],
                      ch)
        
        X = np.ndarray(data_shape)
        for i, c in enumerate(self.camera):
            images = self.log_file.ix[sample_indices][c]
            if self.stack_x:
                images = images.map(lambda fname: read_image(fname.replace("IMG", self.stack_x)))
            else:
                images = images.map(read_image)
            images = images.map(lambda img: pre_processing(img, self.resize_to) if img is not None else np.zeros((data_shape[1], data_shape[2], ch)))
            X[i * self.batch_size : (i+1) * self.batch_size] = np.stack(images)
        self.idx += 1
        return X
    
    def generate_y(self, sample_indices):
        number_of_camera = len(self.camera)
        y = np.ndarray((self.batch_size) * number_of_camera)
        for c, shift, i in zip(self.camera, self.shift_value, range(len(self.camera))):            
            y_slice = y[self.batch_size * i:self.batch_size * (i + 1)]
            y_slice[:] = self.steering[sample_indices]
            
            # shift the steering angle, but prevent sign change
            shifted_y_slice = y_slice + shift
            sign_change = np.dot(shifted_y_slice, y_slice) < 0
            y_slice[:] = shifted_y_slice
            y_slice[sign_change] = 0
                

        assert np.absolute(y).sum() >= 0.0
        return y
        

    def __next__(self):
        sample_indices = np.random.choice(self.log_file.index, size=self.batch_size)
        
        X = self.generate_X(sample_indices)
        y = self.generate_y(sample_indices)
        
        
        return X, y
    
    def __iter__(self):
        return self


class DataGenerator2(object):
    """
    A simplified version of DataGenerator was used for fine tuning.
    This generator do a simple data augumentation by randomly shifting
    the picture horizontally (between -100 and 100 pixels) the steering
    angle is adjusted (the adjusment amount is controlled by #angle_per_pixel)
    """

    def __init__(self, log_file, data_dir=None, batch_size=10, nb_copy=4,
                 target="steering", cameras="center left right".split(),
                 base_shift=[.0, .05, -.05],
                 angle_per_pixel=ANGLE_PER_PIXEL,
                 preprocessing=None):
        self.X = log_file[cameras]
        self.Y = log_file[target]
        self.M = np.float32([[1,0,None],[0,1,0]])
        self.nb_copy = nb_copy
        self.cameras = cameras
        self.base_shift = base_shift
        self.angle_per_pixel = angle_per_pixel
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.preprocessing = preprocessing
#     print(X.shape, Y.shape)

    def __next__(self):
        x_rt = []
        y_rt = []
        for cnt in range(self.batch_size):
            idx = np.random.randint(0, len(self.X))
            x, y = self.X.ix[idx], self.Y.ix[idx]
            if cnt >= self.batch_size:
                raise StopIteration
            for c, shift in zip(self.cameras, self.base_shift):
                fname = x[c] if not self.data_dir else path.join(data_dir, x[c])
                img = plt.imread(fname.strip())
                if self.preprocessing:
                    img = self.preprocessing(img)
                orig_y = y
                y = y + shift
                x_rt.append(img)
                y_rt.append(y)
                for img_shift in np.random.uniform(-100, 100, size=self.nb_copy).astype(np.int):
                    self.M[0][-1] = img_shift
                    dst = cv2.warpAffine(img, self.M, (img.shape[1], img.shape[0]))
                    x_rt.append(dst)
                    y_rt.append(y + img_shift * self.angle_per_pixel)
        return np.array(x_rt), np.array(y_rt)
                    
    def __iter__(self):
        return self


def model_init(input_shape=(66, 200, 3)):

    # Following the architect in Nvidia's paper, I multiplied the depth of Conv2D layers
    # and increase the number of nuerons in Dense layer
    # Adam optimizer was used

    model = Sequential()
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), input_shape=input_shape,
                            activation='relu', bias=True, border_mode="valid"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu', bias=True, border_mode="valid"))
    model.add(Convolution2D(64, 5, 5, subsample=(2,2), activation='relu', bias=True, border_mode="valid"))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1), activation='relu', bias=True, border_mode="valid"))
    model.add(Convolution2D(96, 3, 3, subsample=(1,1), activation='relu', bias=True, border_mode="valid"))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    return model


if __name__ == "__main__":

# first phase training, using udacity's data and lane recovering data recorded by simulator
    log_file = pd.read_csv("data/driving_log.csv", index_col=False)
    log_file['steer'] = log_file.steering
    recover_log_file = pd.read_csv('track1-recover/driving_log.csv',
            names="center left right steer throttle brake speed".split(), index_col=False)
    log_file = pd.concat([recover_log_file, log_file], ignore_index=True)



    model = model_init()
    model.compile(loss='mse', optimizer=Adam(),metrics=['mean_squared_error'])
    
    samples_per_epoch = len(log_file)
    g = DataGenerator(log_file=log_file, batch_size=32, steering_filter=mixed_butter_filter,
                  stack_x="stacked_img", skip_stack=True)
    
    model.fit_generator(g, samples_per_epoch=samples_per_epoch, nb_epoch=1)
    
    model.optimizer.lr /= 10
    model.fit_generator(g, samples_per_epoch=samples_per_epoch, nb_epoch=1)


    validation_log = pd.read_csv("../tmp_img/log_file-01031030.log", sep="\t",
                                    names="fname steering prediction decay throttle".split())
    # del validation_images
    validation_images = np.stack(validation_log.fname.map(read_image).map(lambda img: pre_processing(img, (200, 66))))

    predict_steering = model.predict(validation_images)
#     print("predicted mean: ", predict_# steering.mean(), "std: ", predict_steering.std())
#     print("target mean: ", validation_log.steeriing.mean(), "std: ", predict_steering1.std())
#     p, a = plt.subplots(2)
#     for _a in a:
#         _a.set_ylim((-0.4, 0.4))
#     a[0].plot(predict_steering)
#     a[1].plot(validation_log.steering)
    
    
    
# Fine tuning the model 
# A major problem caused by using keyboard input in the original simulator is that the
# steering angles are very sharp. Leveraging the above model we got 
# (ideally can work with most gentle bends but not on sharp bends) plus a semi-automate 
# version of drive.py, we could record higher quality data more easily. 
# Then using new data for futher training, we could get a model that can automatically 
# drive on the training track.

    # load the newly recorded data
    new_data_log = pd.read_csv("tmp_img/log_file-12210000-very-good.log", sep="\t",
                                names="fname steering prediction decay throttle".split())
    center_images = np.stack(new_data_log.fname.map(read_image)\
                    .map(lambda img: pre_processing(img, (200, 66))))
                
    nb_copy = 6
    sample_per_epoch = len(new_data_log) * nb_copy

    # initiate the simplified generator
    # 6 shifted copies for each picture are generated on the fly
    g = DataGenerator2(validation_log, nb_copy=nb_copy, batch_size=64, base_shift=[.0],
                      cameras=['fname'], preprocessing=lambda img: pre_processing(img, (200, 66)))

    # here we only need to fine tune the model, so I chose a smaller learning rate
    model.optimizer.lr /= 50
    model.fit_generator(g, samples_per_epoch=sample_per_epoch, nb_epoch=1)

    with open("%s.json" % MODEL_NAME, "w") as f:
        f.write(model.to_json())
    model.save_weights("%s.h5" % MODEL_NAME)
