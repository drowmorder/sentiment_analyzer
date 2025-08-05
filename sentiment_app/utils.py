# %%
import joblib
import torch
import os
import re
import nltk

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


cnn_model = None
lr_model = None
tfidf_vectorizer = None 
CNN_tokenizer = None


SUPPORTED_MODELS = ['CNN_model.pt', 'lr_model.joblib', 'rf_model.joblib']

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

def load_models():
    global cnn_model, lr_model, tfidf_vectorizer, CNN_tokenizer

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')

    print(f'Debug: Attempting to load models from {MODEL_DIR}')

    #TFIDF_vectorizer
    try:
        vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
        global tfidf_vectorizer
        tfidf_vectorizer = joblib.load(vectorizer_path)
        print("Vectorizer loaded.")
    except FileNotFoundError as e:
        print(f'tfidf_ectorizer not found: {e}')
    except Exception as e:
        print(f'Error loading vectorizer: {e}')

    #lr_model
    try:
        lr_model_path = os.path.join(MODEL_DIR, 'lr_model.joblib')
        global lr_model
        lr_model = joblib.load(lr_model_path)
        print("LR model loaded.")
    except FileNotFoundError as e:
        print(f'lr_model not found: {e}')
    except Exception as e:
        print(f'Error loading lr_model: {e}')

    #CNN_model
    try:
        cnn_model_path = os.path.join(MODEL_DIR, 'CNN_model.pt')
        global cnn_model
        cnn_model = torch.jit.load(cnn_model_path, map_location=torch.device('cpu'))
        cnn_model.eval()
        print("CNN model loaded.")
    except FileNotFoundError as e:
        print(f'CNN_model not found: {e}')
    except Exception as e:
        print(f'Error loading CNN_model: {e}')

    #CNN_tokenizer
    try:
        cnn_tokenizer_path = os.path.join(MODEL_DIR, 'CNN_tokenizer.joblib')
        global CNN_tokenizer
        CNN_tokenizer = joblib.load(cnn_tokenizer_path)
        print("CNN tokenizer loaded.")
    except FileNotFoundError as e:
        print(f'CNN_tokenizer not found: {e}')
    except Exception as e:
        print(f'Error loading CNN_tokenizer: {e}')  
            

def cleanText(text):
    #remove html tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    #remove punctuation
    text = re.sub(r"[^\w\s']|_", ' ', text)
    
    #remove whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    #convert to lowercase
    text = text.lower()

    #tokenize
    tokens = word_tokenize(text)

    #remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    #lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(lemmatized)

def get_model(model_name: str):
    if model_name == 'CNN_model.pt':
        return cnn_model
    elif model_name == 'lr_model.joblib':
        return lr_model
    else:
        return None