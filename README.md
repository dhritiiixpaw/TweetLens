# 🎀 TweetLens — Human Behavior Sentiment Analysis

TweetLens is a sentiment analysis dashboard built using Python and Streamlit. It analyzes tweets to explore sentiment, emotions, and crime-related categories through interactive visualizations.

This repository contains my **portfolio version of a college group project**, which I further developed and enhanced independently to showcase my learning, technical skills, and project development experience.

## 🌐 Live Demo

[✨ Try TweetLens here!](https://tweetlens-8wcnatwmc4adkrwcd8xqsx.streamlit.app/)

## 📌 Project Background

TweetLens originated as a college group project focused on analyzing human behavior through social media text.

After the academic project, I continued working on it independently to create a portfolio-ready version with additional functionality, an improved interface, and a live deployment.

This version demonstrates my continued exploration of Python, NLP, data visualization, and interactive application development.

## ✨ Features

- **Sentiment Analysis:** Classifies tweets as Positive, Negative, or Neutral.
- **Emotion Detection:** Identifies emotions expressed in tweet text.
- **Crime Classification:** Detects crime-related categories using keyword-based classification.
- **Interactive Dashboard:** Displays sentiment distribution and comparison charts.
- **Emotion Visualization:** Presents emotion analysis results using bar charts.
- **Tweet Activity Timeline:** Visualizes tweet activity over time.
- **WordCloud:** Highlights frequently occurring words in the dataset.
- **Dataset Exploration:** Allows users to explore and filter analyzed tweets.
- **CSV Upload:** Upload custom tweet datasets for analysis.
- **CSV Export:** Download analyzed datasets for further use.

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- Plotly
- Matplotlib
- WordCloud
- Natural Language Processing (NLP)

## 💻 My Portfolio Version

For this portfolio version, I independently worked on extending the application with additional features and improvements, including:

- Adding an interactive tweet activity timeline.
- Integrating a WordCloud visualization.
- Implementing CSV upload and analyzed dataset export.
- Enhancing the dashboard's visual design and usability.
- Deploying the application using Streamlit Community Cloud.
- Preparing the project for public demonstration and portfolio presentation.

## 📊 Project Overview

TweetLens provides an interactive interface for analyzing tweet data and exploring patterns in social media text.

The application processes tweet text and generates sentiment, emotion, and crime-related classifications. These results are presented through interactive charts and data exploration tools.

## 📁 Project Structure

```text
TweetLens/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── tweets.csv
│
└── src/
    ├── sentiment_model.py
    ├── emotion_model.py
    ├── crime_mapper.py
    └── preprocess.py
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd TweetLens
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 📂 Dataset

The project uses a tweet dataset for sentiment and behavioral analysis.

Users can also upload their own CSV files through the application.

For custom datasets, the tweet text column should be named `tweets`.

## 🎯 Project Objective

The objective of TweetLens is to demonstrate how NLP techniques and data visualization can be combined to analyze social media text and explore sentiment, emotions, and crime-related patterns.

## 👩‍💻 Author

**Dhritileena Roy**

GitHub: [@dhritiiixpaw](https://github.com/dhritiiixpaw)

## 📌 Disclaimer

TweetLens is an educational and analytical project. Its sentiment, emotion, and crime-related classifications are predictions or rule-based outputs and should not be treated as definitive judgments about individuals or real-world events.
