from django.shortcuts import render, redirect
from django.http import HttpRequest, HttpResponse
import numpy as np
import torch

from .  import utils

CNN_MAX_SEQUENCE_LENGTH = 256


def index(request: HttpRequest) -> HttpResponse:
    return render(request, 'sentiment_app/index.html')


def analyze(request: HttpRequest, model_name: str) -> HttpResponse:

    review_text = ''
    sentiment_result = None
    rating = None
    error_message = None

    if model_name not in utils.SUPPORTED_MODELS:
        error_message = 'Invalid model name'
        return render(request, 'sentiment_app/analyze.html', {'model_name': model_name, 'error_message': error_message})
    
    if request.method == 'POST':
        review_text = request.POST.get('review_text', '')

        if review_text:
            try:
                selected_model = utils.get_model(model_name)
                
                if selected_model:
                    processed_text = utils.cleanText(review_text)

                    if model_name in ['lr_model.joblib']:
                        text_for_prediction = utils.tfidf_vectorizer.transform([processed_text]).toarray()
                        prediction = selected_model.predict_proba(text_for_prediction)[0]
                        positive_prob = prediction[1]
                        
                    elif model_name == 'CNN_model.pt':
                        if utils.CNN_tokenizer is None:
                            raise ValueError('CNN tokenizer not loaded')
                        words = processed_text.split()
                        token_ids = [utils.CNN_tokenizer.get(word, 0) for word in words]  

                        if len(token_ids) < CNN_MAX_SEQUENCE_LENGTH:
                            token_ids += [0] * (CNN_MAX_SEQUENCE_LENGTH - len(token_ids))
                        else:
                            token_ids = token_ids[:CNN_MAX_SEQUENCE_LENGTH]
                        input_tensor = torch.LongTensor(token_ids).unsqueeze(0)
                        selected_model.eval()
                        with torch.no_grad():
                            output_logits = selected_model(input_tensor)
                            positive_prob = torch.sigmoid(output_logits).item()

                    else:
                        error_message = 'Unsupported model'
                        return render(request, 'sentiment_app/analyze.html', {
                            'model_name': model_name, 
                            'review_text': review_text,
                            'error_message': error_message
                        })


                    if positive_prob > 0.6:
                        sentiment_result = 'Positive'
                        rating = int(positive_prob * 10)
                    elif positive_prob < 0.1:
                        sentiment_result = 'Negative'
                        rating = 1
                    elif positive_prob < 0.4:
                        sentiment_result = 'Negative'
                        rating = int(positive_prob * 10)
                    else:
                        sentiment_result = 'Neutral'
                        rating = 5
                else:
                    error_message = 'Model not loaded or invalid. Check server logs.'
            except Exception as e:
                error_message = f'Error analyzing review: {str(e)}'
        else:
            error_message = 'Please enter a review text.'

    return render(request, 'sentiment_app/analyze.html', {
        'model_name': model_name,
        'review_text': review_text,
        'sentiment_result': sentiment_result,
        'rating': rating,
        'error_message': error_message
    })