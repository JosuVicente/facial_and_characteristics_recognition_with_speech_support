from speech_utils import *
from audio_utils import *
from text_utils import *
from lang_utils import *
from model_utils import *
import cv2
from operator import itemgetter

# Variables
ENCODING_FREQ = 10
encoding_count = 0
last_faces_count = 0
face_encodings = []
predictions = []
font = cv2.FONT_HERSHEY_SIMPLEX

emotion_label_window = []
gender_label_window = []
last_faces = []
label_dict = {'EMOTION': '', 'GENDER': '', 'NAME': '', 'FULL_NAME': ''}

# Language and localization
AVAILABLE_LANGUAGES = ['es','en']
LANGUAGE = 'es'
LANGUAGE_PATH = '../lang/'
AUDIO_PATH = '../audio/'
IMAGE_PATH = '../images/'

lang_helper = Lang_Helper(AVAILABLE_LANGUAGES, LANGUAGE_PATH, AUDIO_PATH, IMAGE_PATH, LANGUAGE)


# Models
model_helper = Model_Helper('../models/face/haarcascade_frontalface_default.xml', 
                            '../models/emotion/simple_CNN.530-0.65.hdf5', 
                            '../models/gender/simple_CNN.81-0.96.hdf5',
                            AUDIO_PATH, IMAGE_PATH)



# Input image 
cv2.namedWindow('main')
video_capture = cv2.VideoCapture(0)


while True:
    predictions = []
    encoding_count += 1
    last_faces_count = len(last_faces)
    last_faces = []
    _, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = model_helper.face_detection.detectMultiScale(gray, 1.3, 5)
      
    do_encode = encoding_count>=ENCODING_FREQ or last_faces_count!=len(faces) 
    
    if (do_encode):
        face_encodings = []
    
    face_index = 0
    
    for (x,y,w,h) in sorted(faces, key=itemgetter(0)):
                
        pred_dict = label_dict.copy();
        
        face_index +=1 
        face = frame[(y - y_offset):(y + h + y_offset),
                    (x - x_offset):(x + w + x_offset)]        
        if (do_encode):
            print('re-encoding')
            face_encodings.append(face_recognition.face_encodings(frame, [tuple([int(y), int(x+w), int(y+h), int(x)])])[0])
            encoding_count = 0
        
        try:
            if (len(face_encodings)>0 & face_index -1 < len(face_encodings)):
                for i in range(len(model_helper.known_faces)):
                    match = face_recognition.compare_faces([model_helper.known_faces[i][2]], face_encodings[face_index-1])
                    if match[0]:
                        pred_dict['NAME'] = model_helper.known_faces[i][0]
                        pred_dict['FULL_NAME'] = model_helper.known_faces[i][1]
                        break;
                  
        except Exception as e: 
            print('*******')
            print(e)
            print('*******')
            continue            
        #print('-----')
        last_faces.append(cv2.cvtColor(face.copy(), cv2.COLOR_RGB2BGR))

        gray_face = gray[(y - y_offset_emotion):(y + h + y_offset_emotion),
                        (x - x_offset_emotion):(x + w + x_offset_emotion)]
        try:
            face = cv2.resize(face, (48, 48))
            gray_face = cv2.resize(gray_face, (48, 48))            
        except:
            continue
        face = np.expand_dims(face, 0)
        face = preprocess_input(face)
        gender_label_arg = np.argmax(model_helper.gender_classifier.predict(face))
        gender = gender_labels[gender_label_arg]
        gender_label_window.append(gender)

        gray_face = preprocess_input(gray_face)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(model_helper.emotion_classifier.predict(gray_face))
        emotion = emotion_labels[emotion_label_arg]
        emotion_label_window.append(emotion)

        if len(gender_label_window) >= frame_window:
            emotion_label_window.pop(0)
            gender_label_window.pop(0)
        try:
            emotion_mode = mode(emotion_label_window)
            gender_mode = mode(gender_label_window)
        except:
            continue
        if gender_mode == gender_labels[0]:
            gender_color = (255, 0, 0)
        else:
            gender_color = (0, 255, 0)   
        
        pred_dict['EMOTION'] = emotion_mode
        pred_dict['GENDER'] = gender_mode
        
        display_text = lang_helper.get_formatted_language_text(pred_dict)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), gender_color, 2)
        cv2.putText(frame, display_text, (x, y - 30), font,
                        .7, gender_color, 1, cv2.LINE_AA)
        
        predictions.append(pred_dict)

    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('main', frame)        
    except:
        continue
    c = chr(cv2.waitKey(2)& 255)
    if (c!= 'ÿ'):
        print(c + " pressed")   
    command = lang_helper.get_command(c.upper())
    if (command == 'language'):
        print('*** Language change *** ')
        lang_helper.switch_to_next_language()
        lang_helper.talk('lang_change')
    elif (command == 'who'):
        print('*** Output predictions selected *** ')
        if (len(predictions) > 0):
            lang_audios =  lang_helper.get_formatted_language_audios(predictions)
            for lang_audio in lang_audios:
                lang_helper.play(lang_audio)
        else:
            lang_helper.talk('no_image')            
    elif (command == 'save'):
        print('*** Save person selected *** ')
        try:
            if (len(last_faces)==1):
                name = '##NONE##'
                while name == '##NONE##':
                    lang_helper.talk('who')
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    else:
                        name = lang_helper.capture_name() 
                        if (name=='##NONE##'):
                            lang_helper.talk('not_understand')                            
                        elif (name == 'cancel'):
                            lang_helper.talk('canceled')                            
                            break
                        else:
                            print('saving face...')
                            full_name = model_helper.save_face(name, lang_helper.current_language, last_faces[0], face_encodings[face_index-1])    
                            print('///////')
                            print(full_name)
                            print(lang_helper.audio_path + 'known/' + full_name + '.mp3')
                            if (full_name!=''):
                                lang_helper.play(lang_helper.audio_path + 'known/' + full_name + '.mp3')
                                lang_helper.talk('saved')                        
                            break                
            elif (len(last_faces)>1):
                lang_helper.talk('more_than_one_face')
            else:
                lang_helper.talk('no_image')
        except:
            continue
    elif (command == 'quit'):
        break
    

video_capture.release()
cv2.destroyAllWindows()

