import gettext

class Text_Helper:    
    def __init__(self, language_path, audio_path, current_language):
        self.language_path = language_path
        self.audio_path = audio_path
        self.current_language = current_language   

        
    def get_():   
        l = gettext.translation('text_', localedir=self.language_path, languages=[self.current_language])
        l.install()
        _ = l.gettext
        return _


    


    
        


