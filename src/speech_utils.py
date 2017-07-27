import speech_recognition as sr
from google.cloud import speech
import io
import os

#######################
GOOGLE_CLOUD_SPEECH_CREDENTIALS_PATH = '../files/GoogleCloudSpeechKey.json'
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

