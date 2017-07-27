import string
from random import *
import os
import glob

import face_recognition


import speech_recognition as sr
from google.cloud import speech
import io
import os

#######################
GOOGLE_CLOUD_SPEECH_CREDENTIALS_PATH = '../files/TFM project-287dc6d9869a.json'
#######################

def transcript_audio(filepath, language, use_cloud):
    transcript = '##NONE##'
    # The name of the audio file to transcribe
    file_name = os.path.join(os.path.dirname(''), filepath)
    
    if use_cloud:
        try:
             # Instantiates a client
            speech_client = speech.Client.from_service_account_json(GOOGLE_CLOUD_SPEECH_CREDENTIALS_PATH)
            
            # Loads the audio into memory
            with io.open(file_name, 'rb') as audio_file:
                content = audio_file.read()
                sample = speech_client.sample(
                    content,
                    source_uri=None,
                    encoding='LINEAR16',
                    sample_rate_hertz=16000)

            # Detects speech in the audio file
            alternatives = sample.recognize(language)
            
            if (len(alternatives)>0):
                transcript = alternatives[0].transcript
        except Exception as e: 
            print(e)
            
    if (transcript == '##NONE##'):
        try:
            r = sr.Recognizer()
            with sr.AudioFile(file_name) as source:
                audio = r.record(source) 
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY", show_all=True)`
            # instead of `r.recognize_google(audio, show_all=True)`
            alternatives = r.recognize_google(audio, show_all=False)
            if (len(alternatives)>0):
                transcript = alternatives
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
       
    return transcript


    
    # Audio Play

import pyaudio
import wave
import time
import sys
import pygame as pg

def play_music(music_file, volume=0.8):
    '''
    stream music with mixer.music module in a blocking manner
    this will stream the sound from disk while playing
    '''
    # set up the mixer
    freq = 44100     # audio CD quality
    bitsize = -16    # unsigned 16 bit
    channels = 2     # 1 is mono, 2 is stereo
    buffer = 2048    # number of samples (experiment to get best sound)
    pg.mixer.init()
    # volume value 0.0 to 1.0
    pg.mixer.music.set_volume(volume)
    clock = pg.time.Clock()
    try:
        pg.mixer.music.load(music_file)
        print("Music file {} loaded!".format(music_file))
    except pg.error:
        print("File {} not found! ({})".format(music_file, pg.get_error()))
        return
    pg.mixer.music.play()
    while pg.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)
        
def play_any_audio(filename):
    pg.mixer.init()
    pg.mixer.music.load(filename)
    pg.mixer.music.play()

def play_audio(filename):    
    WAVE_FILENAME = filename
    if len(sys.argv) < 2:
        print("Plays a wave file.\n\nUsage: %s filename.wav" % WAVE_FILENAME)
        sys.exit(-1)

    wf = wave.open(WAVE_FILENAME, 'rb')

    p = pyaudio.PyAudio()

    def callback(in_data, frame_count, time_info, status):
        data = wf.readframes(frame_count)
        return (data, pyaudio.paContinue)

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True,
                    stream_callback=callback)

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()
    wf.close()

    p.terminate()
    
def record_audio(filename, seconds): 
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    #CHANNELS = 2
    CHANNELS = 1
    #RATE = 44100
    RATE = 16000


    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    
    
import gettext

LANGUAGE_PATH = '../lang/'
LANG = 'es'
LANGUAGE = LANG + '/'
AUDIO_PATH = '../audio/'
KNOWN = 'known/'
TEMP = 'temp/'
IMAGE_PATH = '../images/'
l = gettext.translation('text_', localedir=LANGUAGE_PATH, languages=[LANG])
l.install()
_ = l.gettext


def get_language_audios(path, audios, preds):
    lang_audios = []
    for audio in audios:
        audio_path = path + audio
        for pred in preds:
            audio_path = audio_path.replace('['+pred+']', preds[pred])
        lang_audios.append(audio_path)
    return lang_audios

def get_formatted_language_audios(path, audios, predictions):
    lang_audios = []
    try:
        print(path)
        print(audios)
        print(predictions)
        for prediction in predictions:
            for audio in audios:
                key = audio.split(':')[0]
                if (key == 'GENDER' and prediction['NAME_AUDIO'] != ''):
                    audio_path = AUDIO_PATH + KNOWN + prediction['NAME_AUDIO']
                    lang_audios.append(audio_path)
                else:
                    audio_path = path + audio.split(':')[1]                    
                    for key in prediction:
                        audio_path = audio_path.replace('['+key+']', prediction[key])
                    lang_audios.append(audio_path)
                

    except Exception as e: 
        print('*a******')
        print(e)
        print('*a******')
    return lang_audios

def get_formatted_language_text(language, prediction):
    lang_text = ''
    try:
        text_config = ''
        with open(LANGUAGE_PATH + language + '/text_config.txt') as f:
            for line in f:
                text_config += line.rstrip()
        g = text_config.split(':')[0]
        lang_text = text_config.split(':')[1]
        
        for key in prediction:
            g = g.replace('['+key+']', prediction[key])
            
        l = gettext.translation('text_' + g, localedir=LANGUAGE_PATH, languages=[language])
        l.install()
        __ = l.gettext        
        t = ''
        if (prediction['NAME'] != ''):  
            t = prediction['NAME']
        else:
            if(prediction['GENDER'] != ''):
                t = __(str(prediction['GENDER']))
                
        lang_text = lang_text.replace('[GENDER]', t) 
        t = ''
        if(prediction['EMOTION'] != ''):
            t = __(prediction['EMOTION'])
            
        lang_text = lang_text.replace('[EMOTION]', t)      
    except Exception as e: 
        print('*t******')
        print(e)
        print('*t******')
    return lang_text

config_audios = []
with open(LANGUAGE_PATH+LANGUAGE+'audio_config.txt') as f:
    for line in f:
        config_audios.append(line.rstrip())
        #print(line)
        


label_dict = {'EMOTION': '', 'GENDER': '', 'NAME': '', 'NAME_AUDIO': ''}


pred_test = label_dict.copy();
pred_test['EMOTION'] = 'angry'
pred_test['GENDER'] = 'man'
pred_test['NAME'] = ''
pred_test['NAME_AUDIO'] = ''
text = get_formatted_language_text('es', pred_test)
print(text)


AVAILABLE_LANGUAGES = ['es','en']

def change_language(new_lang):
    _config_audios = []
    with open(LANGUAGE_PATH+new_lang+'/audio_config.txt') as f:
        for line in f:
            _config_audios.append(line.rstrip())
    
    l = gettext.translation('text_', localedir=LANGUAGE_PATH, languages=[new_lang])
    l.install()
    _ = l.gettext  
    return new_lang, _config_audios, _
            
def switch_to_next_language():
    idx = AVAILABLE_LANGUAGES.index(LANG)
    new_idx = 0
    if (idx+1 < len(AVAILABLE_LANGUAGES)):
        new_idx = idx+1
    return change_language(AVAILABLE_LANGUAGES[new_idx])
    
#LANG, config_audios, _ = switch_to_next_language()
#print(LANG)
#print(config_audios)
#print(_('options'))

import cv2
from keras.models import load_model
import numpy as np
from statistics import mode
from utils import preprocess_input
from utils import get_labels

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

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path)
gender_classifier = load_model(gender_model_path)

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

    
from gtts import gTTS
import os
from unidecode import unidecode

def get_language_commands(available_commands):
    l = gettext.translation('text_', localedir=LANGUAGE_PATH, languages=[LANG])
    l.install()
    _ = l.gettext
    commands = []
    for command in available_commands:
        commands.append(_(command))
    return commands

def capture_speech():
    rand = "".join(choice(string.ascii_letters) for x in range(randint(8, 8)))
    temp_wav = AUDIO_PATH + TEMP + rand + '.wav'
    
    #Play beep
    play_music(AUDIO_PATH + 'beep.mp3')
    
    #Record audio
    record_audio(temp_wav, 2)
    play_music(LANGUAGE_PATH + LANGUAGE + 'speech/one_moment.mp3')
    
    #Transcript audio
    transcript = transcript_audio(temp_wav, LANG, True) 
    transcript = unidecode(transcript)
    print('***'+transcript+'***')
    os.remove(temp_wav)
    return transcript.strip()

def capture_selected_command():
    available_commands = ['who','what','save','language','cancel','repeat', 'quit']
    lang_commands = get_language_commands(available_commands)
    transcript = capture_speech()
    if (transcript == '' or transcript == '##NONE##'): 
        return '##NONE##'
    elif (transcript.lower() in lang_commands):
        try:
            return available_commands[lang_commands.index(transcript.lower())]
        except Exception as e: 
            print('**c****')
            print(e)
            print('**c****')
            return '##NONE##'
        return transcript.lower()
    else:
        #return '##UNKNOWN##'
        return '##NONE##'
    

def capture_face_and_name(face):
    available_commands = ['cancel']
    lang_commands = get_language_commands(available_commands)
    
    rand = "".join(choice(string.ascii_letters) for x in range(randint(8, 8)))
    name = "".join(choice(string.ascii_letters) for x in range(randint(6, 6)))

    transcript = capture_speech()    
    transcript = unidecode(transcript)
    
    #if transcript didn't capture anything then exit 
    if (transcript == '' or transcript == '##NONE##'): 
        return None, transcript, None
    #if transcript captures cancelation then cancel
    elif (transcript.lower() in lang_commands):                
        return None, available_commands[lang_commands.index(transcript.lower())], None
    #if transcript ok then proceed
    else:
        mp3_name = transcript + '-' + rand + '.mp3'
        temp_mp3 = AUDIO_PATH + KNOWN + mp3_name
        
        #Convert transcript to standard audio
        tts = gTTS(text=transcript, lang=LANG, slow=False)        
        tts.save(temp_mp3)

        #Play audio back
        play_music(temp_mp3)
        play_music(LANGUAGE_PATH + LANGUAGE + 'speech/saved.mp3')

        #Save face image
        face_img = IMAGE_PATH + KNOWN + transcript + '-' + rand + '.jpg'
        print(face_img)
        #cv2.imshow('image',face)
        cv2.imwrite(face_img, face)

        #Get face encoding
        picture = face_recognition.load_image_file(face_img)        
        face_encoding = face_recognition.face_encodings(picture)[0]
        print('---')
        #print (face_encoding)
        print (transcript)
        print (mp3_name)
        print('---')
        return face_encoding, transcript, mp3_name
    
        
def get_command(c):
    command = '##NONE##'
    if (c==' '):
        try:           
            while command == '##NONE##':
                play_music(LANGUAGE_PATH + LANGUAGE + 'speech/choose_short.mp3')
                nc = chr(cv2.waitKey(2)& 255)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                else:
                    command = capture_selected_command()
                    if (command=='##NONE##'):
                        play_music(LANGUAGE_PATH + LANGUAGE + 'speech/not_understand.mp3')
                    else:                        
                        break                
        except Exception as e: 
            print('*c*****')
            print(e)
            print('*c*****')
            
    if (c=='L' or command=='language'):
        return 'language'
    elif (c=='A' or command=='who'):
        return 'who' 
    elif  (c=='S' or command=='save'):
        return 'save'
    elif (c=='C' or command=='cancel'):
        return 'cancel'
    elif (c=='R' or command=='repeat'):
        return 'repeat'
    if (c=='Q' or command=='quit'):
        return 'quit'
        
        
def update_known_faces(known_faces, name, audio_file_name, face_encoding):
    temp_faces = []
    
    # Remove previous faces with same encoding
    for i in range(len(known_faces)):
        match = face_recognition.compare_faces([known_faces[i][2]], face_encodings[face_index-1])
        if match[0]:
            print(known_faces[i][1] + ' is match')
            image_file = IMAGE_PATH + KNOWN + os.path.splitext(os.path.basename(known_faces[i][1]))[0]+'.jpg'
            audio_file = AUDIO_PATH + KNOWN + known_faces[i][1]
            os.remove(image_file)
            print(image_file + ' deleted')
            os.remove(audio_file)
            print(audio_file + ' deleted')
        else:
            print(known_faces[i][1] + ' no match')
            temp_faces.append(known_faces[i])
        
    # Add new encoding and data to known faces
    temp_faces.append([name, audio_file_name, face_encoding])     
    
    for i in range(len(temp_faces)):
        print(temp_faces[i][0])
        print(temp_faces[i][1])
        #print(known_faces[i][2])

    return temp_faces
    
    
    


# video 
video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.namedWindow('window_frame')
emotion_label_window = []
gender_label_window = []
last_faces = []

ENCODING_FREQ = 10
encoding_count = 0
last_faces_count = 0
face_encodings = []
predictions = []



while True:
    predictions = []
    encoding_count += 1
    last_faces_count = len(last_faces)
    last_faces = []
    _, frame = video_capture.read()
    
    frame_ = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detection.detectMultiScale(gray, 1.3, 5)
      
    do_encode = encoding_count>=ENCODING_FREQ | last_faces_count!=len(faces)
    if (do_encode):
        face_encodings = []
    
    face_index = 0
    for (x,y,w,h) in faces:
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
                for i in range(len(known_faces)):
                    match = face_recognition.compare_faces([known_faces[i][2]], face_encodings[face_index-1])
                    if match[0]:
                        pred_dict['NAME'] = known_faces[i][0]
                        pred_dict['NAME_AUDIO'] = known_faces[i][1]
                        break;
                  
        except Exception as e: 
            print('*******')
            print(e)
            print(len(face_encodings))
            print(face_index)
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
        gender_label_arg = np.argmax(gender_classifier.predict(face))
        gender = gender_labels[gender_label_arg]
        gender_label_window.append(gender)

        gray_face = preprocess_input(gray_face)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
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
        
        display_text = get_formatted_language_text(LANG, pred_dict)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), gender_color, 2)
        cv2.putText(frame, display_text, (x, y - 30), font,
                        .7, gender_color, 1, cv2.LINE_AA)
        #cv2.putText(frame, display_name, (x + 90, y - 30), font,
        #                .7, gender_color, 1, cv2.LINE_AA)
        
        predictions.append(pred_dict)

    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', frame)        
    except:
        continue
    c = chr(cv2.waitKey(2)& 255)
    if (c!= 'ÿ'):
        print(c + " pressed")   
    command = get_command(c.upper())
    if (command == 'language'):
        print('*** Language change *** ')
        LANG, config_audios, _ = switch_to_next_language()
        LANGUAGE = LANG + '/'
        play_music(LANGUAGE_PATH + LANGUAGE + 'speech/lang_change.mp3')
    elif (command == 'who'):
        print('*** Output predictions selected *** ')
        lang_audios =  get_formatted_language_audios(LANGUAGE_PATH + LANGUAGE, config_audios, predictions)
        for lang_audio in lang_audios:
            print(lang_audio)
            play_music(lang_audio)
    elif (command == 'save'):
        print('*** Save person selected *** ')
        try:
            if (len(last_faces)==1):
                name = '##NONE##'
                while name == '##NONE##':
                    play_music(LANGUAGE_PATH + LANGUAGE + 'speech/who.mp3')
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    else:
                        face_encoding, name, audio_file_name = capture_face_and_name(last_faces[0])                        
                        if (name=='##NONE##'):
                            play_music(LANGUAGE_PATH + LANGUAGE + 'speech/not_understand.mp3')
                        elif (name == 'canceled'):
                            play_music(LANGUAGE_PATH + LANGUAGE + 'speech/canceled.mp3')
                            break
                        else:
                            print('update known faces')
                            known_faces = update_known_faces(known_faces, name, audio_file_name, face_encoding)                                 
                            break                
            else:
                play_music(LANGUAGE_PATH + LANGUAGE + 'speech/more_than_one_face.mp3')
        except:
            continue
    elif (command == 'quit'):
        break
    

video_capture.release()
cv2.destroyAllWindows()

