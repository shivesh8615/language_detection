import pickle
import re
from pathlib import Path


__version__="1.0"


with open("langaue_detector-1.0.pkl","rb") as f:
     model = pickle.load(f)

classes = ['Arabic', 'Danish', 'Dutch', 'English', 'French', 'German',
       'Greek', 'Hindi', 'Italian', 'Kannada', 'Malayalam', 'Portugeese',
       'Russian', 'Spanish', 'Sweedish', 'Tamil', 'Turkish']
    
def predict_pipeline(text):
    text = re.sub(r'[!@#$(),\n^%*?\:;~`0-9]','',text)
    text = re.sub(r'[[]]','',text)
    text = text.lower()
    pred = model.predict([text])
    return classes[pred[0]]
