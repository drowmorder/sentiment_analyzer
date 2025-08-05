# IMDb Movie Review Sentiment Analyzer

## A web application with built-in machine learning models to classify movie review sentiment and provide a rating.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-4.x-green?style=for-the-badge&logo=django&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x%2B-orange?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x%2B-red?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

### Table of Contents
- [English](#english)
    - [Project Overview](#project-overview)
    - [Features](#features)
    - [Models Used](#models-used)
    - [Data & Preprocessing](#data--preprocessing)
    - [Installation & Setup](#installation--setup)
    - [Usage](#usage)
    - [Screenshots](#screenshots)
    - [Project Structure](#project-structure)
    - [Additional Resources](#additional-resources)
    - [Future Enhancements](#future-enhancements)
    - [License](#license)
    - [Contact](#contact)
- [Русский](#русский)
    - [Обзор проекта](#обзор-проекта)
    - [Возможности](#возможности)
    - [Используемые модели](#используемые-модели)
    - [Данные и предобработка](#данные-и-предобработка)
    - [Установка и запуск](#установка-и-запуск)
    - [Использование](#использование)
    - [Скриншоты](#скриншоты-1)
    - [Структура проекта](#структура-проекта-1)
    - [Дополнительные ресурсы](#дополнительные-ресурсы)
    - [Планы на будущее](#планы-на-будущее)
    - [Лицензия](#лицензия-1)
    - [Контакты](#контакты)

---

## English

### Project Overview
This project is a web-based sentiment analyzer specifically designed for IMDb movie reviews. It provides users with an intuitive interface to input review text, receive an instant sentiment classification (Positive, Negative, or Neutral), and a corresponding rating on a scale of 1 to 10. The application serves as a practical demonstration of integrating diverse machine learning models (both traditional and deep learning) into a user-friendly Django web application.

### Features
* **Interactive Web Interface:** A straightforward and intuitive front-end for submitting movie reviews.
* **Multiple Model Selection:** Users can choose between a Logistic Regression model and a Convolutional Neural Network (CNN) for sentiment prediction, with more models planned for future integration.
* **Sentiment Classification:** Categorizes review text into one of three sentiments: **Positive**, **Negative**, or **Neutral**.
* **1-10 Rating System:** Provides a normalized rating based on the predicted sentiment score, calculated using a formula derived from the model's `predict_proba` output:
    * Rating > 6: Positive
    * Rating = 5: Neutral
    * Rating < 4: Negative
* **Robust Preprocessing:** Handles essential text cleaning, tokenization, padding, and vocabulary mapping for optimal model input.
* **Scalable Architecture:** Designed for easy integration and expansion with additional machine learning models.

### Models Used
The application currently utilizes two distinct machine learning models for sentiment analysis:

1.  **Logistic Regression:**
    * A highly efficient and interpretable linear model.
    * This model is built upon **TF-IDF (Term Frequency-Inverse Document Frequency)** features, effectively capturing the importance of words within the review relative to the entire dataset.
    * Provides a strong and fast baseline for sentiment prediction.

2.  **Convolutional Neural Network (CNN):**
    * A powerful deep learning architecture particularly effective for text classification, capable of capturing local patterns (n-grams) within sentences.
    * It leverages **pre-trained GloVe Word Embeddings (6 Billion tokens, 100 dimensions)** to convert words into rich, dense vector representations, allowing the model to understand semantic similarities and relationships between words.
    * The CNN architecture includes parallel 1D convolutional layers with varying `kernel_sizes` (2, 3, and 4) to detect different n-gram patterns. Each convolutional block uses 256 filters.
    * A `dropout_rate` of 0.5 is applied to the final feature representation to prevent overfitting.
    * Excels at capturing complex dependencies and provides high accuracy.

### Data & Preprocessing
The models were trained and evaluated using the **IMDb Large Movie Review Dataset**, publicly available on Kaggle: [https://www.kaggle.com/datasets/pranayprasad/aclimdb](https://www.kaggle.com/datasets/pranayprasad/aclimdb).

The text preprocessing pipeline is crucial for preparing raw review text for model input. It includes:
* Lowercasing all text.
* Removing HTML tags, punctuation, and special characters.
* Tokenization (splitting text into individual words).
* Building a vocabulary that maps unique words to numerical IDs.
* Padding (or truncating) sequences to a fixed `max_length` of **256** tokens to ensure consistent input dimensions for the models.

### Installation & Setup
Follow these steps to get the project up and running locally:

1.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPO_URL_HERE]
    cd IMDb-Movie-Review-Sentiment-Analyzer # Or whatever your repo folder name is
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate   # On Windows (use git bash or similar for 'source')
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download GloVe Embeddings:**
    The CNN model requires specific pre-trained GloVe embeddings (`glove.6B.100d.txt`).
    * Download `glove.6B.zip` from the [GloVe website (Stanford NLP)](https://nlp.stanford.edu/projects/glove/).
    * Extract the `glove.6B.100d.txt` file from the downloaded zip.
    * **Place the `glove.6B.100d.txt` file into the `data/` directory within your cloned project.** (e.g., `[YOUR_REPO_NAME]/data/glove.6B.100d.txt`)

5.  **Run Django Migrations:**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

6.  **Start the Django Development Server and Open Application:**
    * **Option A (Recommended for Windows): Use the provided batch script:**
        After completing the above steps, simply double-click the `start_server.bat` file located in the project root. This script will automatically start the Django development server and open the application's starting page in your default web browser.
    * **Option B (Manual): Run the server and open browser separately:**
        ```bash
        python manage.py runserver
        ```
        Then, open your web browser and navigate to `http://127.0.0.1:8000/sentiment`.

### Usage
1.  On the initial landing page, you will be prompted to select your preferred sentiment analysis model (Logistic Regression or CNN).
2.  After selecting a model, you will be redirected to the analysis page.
3.  Enter the movie review text into the provided text area.
4.  Click the "Analyze" button to instantly receive the predicted sentiment and the corresponding 1-10 rating.

### Screenshots (incoming)
* **Recommendation:** This section is crucial! **Please replace this text with high-quality screenshots or a GIF demonstrating your app.**
* **What to include:**
    * A screenshot of the initial model selection page.
    * A screenshot of the analysis input page (with an example review typed in).
    * A screenshot of the results page, clearly showing the sentiment and rating.
    * (Optional but great) A short GIF showing the flow from input to output for one review.

### Project Structure
.

├── IMDb-Movie-Review-Sentiment-Analyzer/<br/>
│   ├── sentiment_app/         # Django app: views, URLs, templates, ML integration logic<br/>
│   │   ├── migrations/<br/>
│   │   ├── templates/         # HTML templates for the web interface<br/>
│   │   ├── static/            # Static files (CSS, JS, images)<br/>
│   │   ├── init.py<br/>
│   │   ├── admin.py<br/>
│   │   ├── apps.py<br/>
│   │   ├── models.py<br/>
│   │   ├── urls.py<br/>
│   │   ├── views.py<br/>
│   │   └── utils.py          # (e.g., contains text preprocessing functions, rating calculation)<br/>
│   ├── [YOUR_DJANGO_PROJECT_ROOT_FOLDER_NAME]/ # Django project settings folder (e.g., mysite/ or config/)<br/>
│   │   ├── init.py<br/>
│   │   ├── asgi.py<br/>
│   │   ├── settings.py<br/>
│   │   ├── urls.py<br/>
│   │   └── wsgi.py<br/>
│   ├── data/                 # Folder for datasets, vocabulary, and GloVe embeddings<br/>
│   │   ├── glove.6B.100d.txt<br/>
│   │   └── # other data files like vocabulary, etc.<br/>
│   ├── models/               # Folder for saved ML models (.joblib for LR, .pt for CNN)<br/>
│   │   ├── lr_model.joblib<br/>
│   │   └── CNN_model.pt<br/>
│   ├── manage.py             # Django project entry point<br/>
│   ├── requirements.txt      # Python dependencies list<br/>
│   ├── start_server.bat      # Convenience script for Windows users to start server and open browser<br/>
│   └── .gitignore            # Files/folders ignored by Git<br/>
└── README.md<br/>

### Additional Resources
* **Exploratory Data Analysis (EDA) Notebook:** Explore the initial data analysis, preprocessing steps, and insights in the `data/SA_EDA/SentimentAnalysis.ipynb` Jupyter Notebook. This provides a deeper dive into the dataset and feature engineering.

### Future Enhancements
* Integrate more advanced NLP models, such as Transformer-based models (e.g., BERT, RoBERTa).
* Implement real-time review fetching from IMDb or other movie databases.
* Expand the sentiment analysis capabilities to other domains (e.g., product reviews, social media posts).
* Develop user authentication and a history feature for past analyses.
* Containerize the application using Docker for easier deployment.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Contact
* **Kozhukhovskiy Dmitriy**
* **Links:** [Your link URL]

---

<br>
<br>

---

## Русский

### Обзор проекта
Этот проект представляет собой веб-анализатор настроений, специально разработанный для рецензий фильмов IMDb. Он предоставляет пользователям интуитивно понятный интерфейс для ввода текста рецензии, получения мгновенной классификации настроения (Положительное, Отрицательное или Нейтральное) и соответствующей оценки по шкале от 1 до 10. Приложение служит практической демонстрацией интеграции различных моделей машинного обучения (как традиционных, так и глубоких) в удобное веб-приложение на базе Django.

### Возможности
* **Интерактивный веб-интерфейс:** Простой и интуитивно понятный интерфейс для отправки рецензий на фильмы.
* **Выбор модели:** Пользователи могут выбирать между моделью логистической регрессии и сверточной нейронной сетью (CNN) для прогнозирования настроения, с планами по интеграции большего количества моделей в будущем.
* **Классификация настроения:** Категоризирует текст рецензии на одно из трех настроений: **Положительное**, **Отрицательное** или **Нейтральное**.
* **Система оценки 1-10:** Предоставляет нормализованную оценку на основе прогнозируемой оценки настроения, рассчитанную по формуле, полученной из выходных данных `predict_proba` модели:
    * Оценка > 6: Положительная
    * Оценка = 5: Нейтральная
    * Оценка < 4: Отрицательная
* **Надежная предобработка:** Обрабатывает необходимую очистку текста, токенизацию, заполнение и сопоставление со словарем для оптимального ввода в модель.
* **Масштабируемая архитектура:** Разработана для легкой интеграции и расширения с помощью дополнительных моделей машинного обучения.

### Используемые модели
В настоящее время приложение использует две различные модели машинного обучения для анализа настроений:

1.  **Логистическая регрессия:**
    * Высокоэффективная и интерпретируемая линейная модель.
    * Эта модель построена на признаках **TF-IDF (частота термов-обратная частота документов)**, эффективно отражая важность слов в рецензии относительно всего набора данных.
    * Обеспечивает надежную и быструю базовую линию для прогнозирования настроения.

2.  **Сверточная нейронная сеть (CNN):**
    * Мощная архитектура глубокого обучения, особенно эффективная для классификации текста, способная улавливать локальные паттерны (n-граммы) в предложениях.
    * Использует **предварительно обученные словесные эмбеддинги GloVe (6 миллиардов токенов, 100 измерений)** для преобразования слов в богатые, плотные векторные представления, что позволяет модели понимать семантические сходства и отношения между словами.
    * Архитектура CNN включает параллельные одномерные сверточные слои с различными размерами ядер (2, 3 и 4) для обнаружения различных n-граммных паттернов. Каждый сверточный блок использует 256 фильтров.
    * Применяется `dropout_rate` 0.5 к конечному представлению признаков для предотвращения переобучения.
    * Отлично справляется с улавливанием сложных зависимостей и обеспечивает высокую точность.

### Данные и предобработка
Модели были обучены и оценены с использованием **набора данных больших рецензий фильмов IMDb**, общедоступного на Kaggle: [https://www.kaggle.com/datasets/pranayprasad/aclimdb](https://www.kaggle.com/datasets/pranayprasad/aclimdb).

Пайплайн предобработки текста имеет решающее значение для подготовки необработанного текста рецензии для ввода в модель. Он включает:
* Приведение всего текста к нижнему регистру.
* Удаление HTML-тегов, знаков препинания и специальных символов.
* Токенизацию (разделение текста на отдельные слова).
* Создание словаря, который сопоставляет уникальные слова с числовыми идентификаторами.
* Заполнение (или усечение) последовательностей до фиксированной `max_length` **256** токенов для обеспечения согласованных входных измерений для моделей.

### Установка и запуск
Выполните следующие шаги, чтобы запустить проект локально:

1.  **Клонируйте репозиторий:**
    ```bash
    git clone [ВАШ_URL_РЕПОЗИТОРИЯ]
    cd IMDb-Movie-Review-Sentiment-Analyzer # Или название папки вашего репозитория
    ```

2.  **Создайте и активируйте виртуальное окружение:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Для Linux/macOS
    # venv\Scripts\activate   # Для Windows (используйте git bash или аналогичный для 'source')
    ```

3.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Загрузите эмбеддинги GloVe:**
    Модель CNN требует определенных предварительно обученных эмбеддингов GloVe (`glove.6B.100d.txt`).
    * Загрузите `glove.6B.zip` с [веб-сайта GloVe (Stanford NLP)](https://nlp.stanford.edu/projects/glove/).
    * Распакуйте файл `glove.6B.100d.txt` из загруженного архива.
    * **Поместите файл `glove.6B.100d.txt` в директорию `data/` внутри вашего клонированного проекта.** (например, `[ИМЯ_ВАШЕГО_РЕПОЗИТОРИЯ]/data/glove.6B.100d.txt`)

5.  **Выполните миграции Django:**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

6.  **Запустите сервер разработки Django и откройте приложение:**
    * **Вариант А (Рекомендуется для Windows): Используйте предоставленный batch-скрипт:**
        После выполнения вышеуказанных шагов просто дважды щелкните файл `start_server.bat`, расположенный в корне проекта. Этот скрипт автоматически запустит сервер разработки Django и откроет стартовую страницу приложения в вашем браузере по умолчанию.
    * **Вариант Б (Вручную): Запустите сервер и откройте браузер отдельно:**
        ```bash
        python manage.py runserver
        ```
        Затем откройте веб-браузер и перейдите по адресу `http://127.0.0.1:8000/`.

### Использование
1.  На начальной странице вам будет предложено выбрать предпочитаемую модель анализа настроений (Логистическая регрессия или CNN).
2.  После выбора модели вы будете перенаправлены на страницу анализа.
3.  Введите текст рецензии фильма в предоставленную текстовую область.
4.  Нажмите кнопку "Анализировать", чтобы мгновенно получить прогнозируемое настроение и соответствующую оценку от 1 до 10.

### Скриншоты(в работе)
* **Рекомендация:** Этот раздел очень важен! **Пожалуйста, замените этот текст высококачественными скриншотами или GIF-анимацией, демонстрирующими ваше приложение.**
* **Что включить:**
    * Скриншот начальной страницы выбора модели.
    * Скриншот страницы ввода для анализа (с введенным примером рецензии).
    * Скриншот страницы результатов, четко показывающий настроение и оценку.
    * (Необязательно, но очень хорошо) Короткая GIF-анимация, показывающая поток от ввода до вывода для одной рецензии.

### Структура проекта
.

├── IMDb-Movie-Review-Sentiment-Analyzer/<br/>
│   ├── sentiment_app/         # Приложение Django: представления, URL-адреса, шаблоны, логика интеграции ML<br/>
│   │   ├── migrations/<br/>
│   │   ├── templates/         # HTML-шаблоны для веб-интерфейса<br/>
│   │   ├── static/            # Статические файлы (CSS, JS, изображения)<br/>
│   │   ├── init.py<br/>
│   │   ├── admin.py<br/>
│   │   ├── apps.py<br/>
│   │   ├── models.py<br/>
│   │   ├── urls.py<br/>
│   │   ├── views.py<br/>
│   │   └── utils.py          # (например, содержит функции предобработки текста, расчет рейтинга)<br/>
│   ├── [ИМЯ_ВАШЕЙ_КОРНЕВОЙ_ПАПКИ_ПРОЕКТА_DJANGO]/ # Папка настроек проекта Django (например, mysite/ или config/)<br/>
│   │   ├── init.py<br/>
│   │   ├── asgi.py<br/>
│   │   ├── settings.py<br/>
│   │   ├── urls.py<br/>
│   │   └── wsgi.py<br/>
│   ├── data/                 # Папка для наборов данных, словаря и эмбеддингов GloVe<br/>
│   │   ├── glove.6B.100d.txt<br/>
│   │   └── # другие файлы данных, такие как словарь и т. д.<br/>
│   ├── models/               # Папка для сохраненных моделей ML (.joblib для LR, .pt для CNN)<br/>
│   │   ├── lr_model.joblib<br/>
│   │   └── CNN_model.pt<br/>
│   ├── manage.py             # Точка входа проекта Django<br/>
│   ├── requirements.txt      # Список зависимостей Python<br/>
│   ├── start_server.bat      # Удобный скрипт для пользователей Windows для запуска сервера и открытия браузера<br/>
│   └── .gitignore            # Файлы/папки, игнорируемые Git<br/>
└── README.md<br/>

### Дополнительные ресурсы
* **Ноутбук с EDA (исследовательским анализом данных):** Изучите первоначальный анализ данных, шаги предобработки и выводы в Jupyter Notebook `data/SA_EDA/SentimentAnalysis.ipynb`. Это дает более глубокое представление о наборе данных и инженерии признаков.

### Планы на будущее
* Интеграция более продвинутых моделей NLP, таких как модели на основе Трансформеров (например, BERT, RoBERTa).
* Реализация получения рецензий в реальном времени с IMDb или других баз данных фильмов.
* Расширение возможностей анализа настроений на другие области (например, рецензии на товары, сообщения в социальных сетях).
* Разработка аутентификации пользователя и функции истории для прошлых анализов.
* Контейнеризация приложения с использованием Docker для упрощения развертывания.

### Лицензия
Этот проект распространяется под лицензией MIT — подробности см. в файле [LICENSE](LICENSE).

### Контакты
* **Кожуховский Дмитрий
* **
* **Links:** 
