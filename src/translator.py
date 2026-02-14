from deep_translator import GoogleTranslator

class WordTranslator:
    def __init__(self):
        self.current_word = ""
    
    def append_letter(self, idx):
        letter = chr(ord('A') + int(idx))
        self.current_word += letter
        return letter
    
    def translate_to_spanish(self):
        if not self.current_word: return "Empty"
        return GoogleTranslator(source='en', target='es').translate(self.current_word)
    
    def reset(self):
        self.current_word = ""