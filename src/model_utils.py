import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imresize
from random import shuffle
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
from statistics import mode
import glob
import os
import face_recognition
import string 
from random import *
from gtts import gTTS

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0:'angry',1:'disgust',2:'sad',3:'happy',
                    4:'sad',5:'surprise',6:'neutral'}
    elif dataset_name == 'imdb':
        return {0:'woman', 1:'man'}
    else:
        raise Exception('Invalid dataset name')

def preprocess_input(images):
    images = images/255.0
    return images

def _imread(image_name):
        return imread(image_name)

def _imresize(image_array, size):
        return imresize(image_array, size)

def split_data(ground_truth_data, training_ratio=.8, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle == True:
        shuffle(ground_truth_keys)
    num_train = int(round(training_ratio * len(ground_truth_keys)))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys

def display_image(image_array):
    image_array =  np.squeeze(image_array).astype('uint8')
    plt.imshow(image_array)
    plt.show()

def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical


# parameters
detection_model_path = '../models/face/haarcascade_frontalface_default.xml'
emotion_model_path = '../models/emotion/simple_CNN.530-0.65.hdf5'
gender_model_path = '../models/gender/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
frame_window = 10
x_offset_emotion = 20
y_offset_emotion = 40
x_offset = 30
y_offset = 60

class Model_Helper:    
    def __init__(self, detection_model_path, emotion_model_path, current_language, audio_path, image_path):
        self.audio_path = audio_path
        self.image_path = image_path

        print('Loading gender detector...')
        self.gender_classifier = load_model(gender_model_path)
        
        print('Loading face detector...')
        self.face_detection = cv2.CascadeClassifier(detection_model_path)
        
        print('Loading emotion detector...')
        self.emotion_classifier = load_model(emotion_model_path)  

        print('Loading known faces...')

        self.known_faces = []

        for filepath in glob.iglob(self.image_path + 'known/*.*', recursive=True):  
            try:
                filename = os.path.splitext(os.path.basename(filepath))[0]
                name = os.path.splitext(filename)[0].split('-')[0]
                picture = face_recognition.load_image_file(filepath)
                encoding = face_recognition.face_encodings(picture)[0]
                self.known_faces.append([name, filename, encoding])
            except Exception as e: 
                try:
                    os.remove(self.image_path + 'known/' + filename+'.jpg')
                    os.remove(self.audio_path + 'known/' + filename+'.mp3')
                except Exception as e: 
                    print(e)                    
            

        print(str(len(self.known_faces)) + ' faces loaded')

    def update_known_faces(self, name, audio_file_name, face_encoding, current_encoding):
        temp_faces = []
        
        # Remove previous faces with same encoding
        for i in range(len(self.known_faces)):
            match = face_recognition.compare_faces([self.known_faces[i][2]], current_encoding)
            if match[0]:
                print(self.known_faces[i][1] + ' is match')
                image_file = self.image_path + 'known/' + self.known_faces[i][1]+'.jpg'
                audio_file = self.audio_path + 'known/' + self.known_faces[i][1]+'.mp3'
                os.remove(image_file)
                print(image_file + ' deleted')
                os.remove(audio_file)
                print(audio_file + ' deleted')
            else:
                print(self.known_faces[i][1] + ' no match')
                temp_faces.append(self.known_faces[i])
        # Add new encoding and data to known faces
        temp_faces.append([name, audio_file_name, face_encoding])     
        print(name + ' added')
        self.known_faces = temp_faces      

    def save_face(self, name, language, face, current_encoding):
        try:
            rand = "".join(choice(string.ascii_letters) for x in range(randint(8, 8)))
            full_name = name + '-' + rand
            path_audio = self.audio_path + 'known/' + full_name + '.mp3'
            path_image = self.image_path + 'known/' + full_name + '.jpg'
            
            #Convert transcript to standard audio
            tts = gTTS(text=name, lang=language, slow=False)        
            tts.save(path_audio)
            
            #cv2.imshow('image',face)
            cv2.imwrite(path_image, face)
            
            #Get face encoding
            picture = face_recognition.load_image_file(path_image)        
            face_encoding = face_recognition.face_encodings(picture)[0]

            self.update_known_faces(name, full_name, face_encoding, current_encoding)
            return full_name
        except Exception as e: 
            print('**s****')
            print(e)
            print('**s****')
            return ''

            
        


     
        

