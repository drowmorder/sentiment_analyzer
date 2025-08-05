from django.apps import AppConfig


class SentimentAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'sentiment_app'

    def ready(self):
        from . import utils
        try:
            utils.load_models()
        except Exception as e:
            print(f"Failed to load machine learning models: {e}")
            # raise e # Uncomment this to prevent server startup if models don't load
