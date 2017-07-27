from gtts import gTTS
import os
from unidecode import unidecode
import gettext
from audio_utils import *
import cv2
from random import *
from speech_utils import *
import string

class Lang_Helper:


     
    def __init__(self, available_languages, language_path, audio_path, image_path, current_language):
        self.available_languages = available_languages
        self.language_path = language_path
        self.audio_path = audio_path
        self.audio_path = audio_path
        self.current_language = current_language   
        self.config_audios = []
        with open(self.language_path + self.current_language + '/audio_config.txt') as f:
            for line in f:
                self.config_audios.append(line.rstrip())
        l = gettext.translation('text_', localedir=self.language_path, languages=[self.current_language])
        l.install()
        self._ = l.gettext  

    def change_language(self, new_lang):
        self.config_audios = []
        with open(self.language_path+new_lang+'/audio_config.txt') as f:
            for line in f:
                self.config_audios.append(line.rstrip())
        
        l = gettext.translation('text_', localedir=self.language_path, languages=[new_lang])
        l.install()
        self._ = l.gettext  
        self.current_language = new_lang        
                
    def switch_to_next_language(self):
        idx = self.available_languages.index(self.current_language)
        new_idx = 0
        if (idx+1 < len(self.available_languages)):
            new_idx = idx+1
        self.change_language(self.available_languages[new_idx])
        

    def get_language_commands(self, available_commands):
        l = gettext.translation('text_', localedir=self.language_path, languages=[self.current_language])
        l.install()
        _ = l.gettext
        commands = []
        for command in available_commands:
            commands.append(_(command))
        return commands

    def capture_speech(self):
        rand = "".join(choice(string.ascii_letters) for x in range(randint(8, 8)))
        temp_wav = self.audio_path + 'temp/' + rand + '.wav'
        
        #Play beep
        self.play(self.audio_path + 'beep.mp3')
        
        #Record audio
        record_audio(temp_wav, 2)
        self.talk('one_moment')
        
        #Transcript audio
        transcript = transcript_audio(temp_wav, self.current_language, True) 
        transcript = unidecode(transcript)
        print('***'+transcript+'***')
        os.remove(temp_wav)
        return transcript.strip()

    def capture_selected_command(self):
        available_commands = ['who','what','save','language','cancel','repeat', 'quit','options', 'keys', 'repeat']
        lang_commands = self.get_language_commands(available_commands)
        transcript = self.capture_speech()
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

    def capture_custom_command(self, available_commands):
        lang_commands = self.get_language_commands(available_commands)
        transcript = self.capture_speech()
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
        

    def capture_name(self):
        available_commands = ['cancel']
        lang_commands = self.get_language_commands(available_commands)        
        
        transcript = self.capture_speech()    
        transcript = unidecode(transcript)
        
        #if transcript didn't capture anything then exit 
        if (transcript == '' or transcript == '##NONE##'): 
            return '##NONE##'
        #if transcript captures cancelation then cancel
        elif (transcript.lower() in lang_commands):                
            return available_commands[lang_commands.index(transcript.lower())]
        #if transcript ok then proceed
        else:
            return transcript
            
        
    def get_command(self, c):
        command = '##NONE##'
        if (c==' '):
            try:           
                while command == '##NONE##':
                    self.talk('choose_short')
                    nc = chr(cv2.waitKey(2)& 255)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    else:
                        command = self.capture_selected_command()
                        if (command=='##NONE##'):
                            self.talk('not_understand')
                            self.talk('repeat_options')
                            new_command = '##NONE##'
                            new_command = self.capture_custom_command(['yes', 'no'])
                            if (new_command == 'yes'):
                                self.talk('commands')
                            else:
                                self.talk('ok')
                                command = 'cancel'
                                break;
                        else:                        
                            break                
            except Exception as e: 
                print('*c*****')
                print(e)
                print('*c*****')
        elif (c=='0'):
            try:           
                while command == '##NONE##':
                    self.talk('choose')
                    nc = chr(cv2.waitKey(2)& 255)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    else:
                        new_command = '##NONE##'
                        new_command = self.capture_custom_command((['commands', 'keys', 'cancel']))
                        if (new_command=='##NONE##'):
                            self.talk('not_understand')
                            self.talk('commands')
                            return self.get_command(' ')
                        elif (new_command=='commands'):
                            self.talk('commands')
                            return self.get_command(' ')
                        elif (new_command=='keys'):
                            self.talk('keys')
                            return self.get_command(' ') 
                        elif (new_command=='cancel'):
                            return 'cancel'                     
                        else :                        
                            self.talk('not_understand')
                            self.talk('commands')
                            return self.get_command(' ')
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
            self.talk('canceled')
            return 'cancel'
        elif (command=='repeat'):
            self.talk('commands')
            return self.get_command(' ')        
        if (c=='Q' or command=='quit'):
            return 'quit'

    def get_language_audios(path, audios, preds):
        lang_audios = []
        for audio in audios:
            audio_path = path + audio
            for pred in preds:
                audio_path = audio_path.replace('['+pred+']', preds[pred])
            lang_audios.append(audio_path)
        return lang_audios

    def get_formatted_language_audios(self, predictions):
        lang_audios = []
        try:
            print(predictions)
            for prediction in predictions:
                for audio in self.config_audios:
                    key = audio.split(':')[0]
                    if (key == 'GENDER' and prediction['FULL_NAME'] != ''):
                        audio_path = self.audio_path + 'known/' + prediction['FULL_NAME'] + '.mp3'
                        lang_audios.append(audio_path)
                    else:
                        audio_path = self.language_path + self.current_language + '/' + audio.split(':')[1]                    
                        for key in prediction:
                            audio_path = audio_path.replace('['+key+']', prediction[key])
                        lang_audios.append(audio_path)
                    

        except Exception as e: 
            print('*a******')
            print(e)
            print('*a******')
        return lang_audios

    def get_formatted_language_text(self, prediction):
        lang_text = ''
        try:
            text_config = ''
            with open(self.language_path + self.current_language + '/text_config.txt') as f:
                for line in f:
                    text_config += line.rstrip()
            g = text_config.split(':')[0]
            lang_text = text_config.split(':')[1]
            
            for key in prediction:
                g = g.replace('['+key+']', prediction[key])
                
            l = gettext.translation('text_' + g, localedir=self.language_path, languages=[self.current_language])
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

    def talk(self, audio_name):
        self.play(self.language_path + self.current_language + '/speech/' + audio_name + '.mp3')

    def play(self, audio_path):
        play_audio(audio_path)        
