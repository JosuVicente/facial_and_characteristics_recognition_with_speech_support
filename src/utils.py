import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imresize
from random import shuffle
import numpy as np

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

known_faces = []


for filepath in glob.iglob('../images/known/*.*', recursive=True):  
    filename = os.path.splitext(os.path.basename(filepath))[0]+'.mp3'
    name = os.path.splitext(filename)[0].split('-')[0]
    picture = face_recognition.load_image_file(filepath)
    encoding = face_recognition.face_encodings(picture)[0]
    known_faces.append([name, filename, encoding])

for i in range(len(known_faces)):
    print(known_faces[i][0])
    print(known_faces[i][1])
    #print(known_faces[i][2])
