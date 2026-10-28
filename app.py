from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import plotly.express as px

from src.sentiment import analyze_sentiment
from src.emotion import detect_emotion
from src.crime_mapper import classify_crime_multilabel


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="TweetLens",
    page_icon="🎀",
    layout="wide"
)
# ---------------- PASTEL UPLOADER & DATASET PREVIEW ----------------

st.markdown("""
<style>

/* Choose Dataset - File Uploader */
[data-testid="stFileUploader"] {
    background-color: #E3F2FD;
    border: 2px dashed #A8C9E8;
    border-radius: 15px;
    padding: 15px;
}

/* Browse Files Button */
[data-testid="stFileUploader"] button {
    background-color: #C8DCC8 !important;
    color: #405B45 !important;
    border: none !important;
    border-radius: 10px !important;
}

/* Dataset Preview Table */
[data-testid="stDataFrame"] {
    background-color: #E3F2FD;
    border-radius: 12px;
    overflow: hidden;
}

/* Dataset Preview Header */
[data-testid="stDataFrame"] th {
    background-color: #C8DCC8 !important;
    color: #405B45 !important;
}

/* Dataset Preview Text */
[data-testid="stDataFrame"] td {
    background-color: #F0F7FF !important;
    color: #526B55 !important;
}

/* File Uploader Text */
[data-testid="stFileUploader"] label {
    color: #526B55 !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------- CUSTOM THEME ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff7f5 0%, #f1f8ef 100%);
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
}

h1 {
    color: #476b50 !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #527b5c !important;
}

p, li {
    color: #526B85;
    font-size: 16px;
    font-weight: 500;
}

[data-testid="stMetric"] {
    background-color: #fffdfb;
    border: 1px solid #f3cbd2;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 5px 15px rgba(180, 130, 140, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #8a7180 !important;
}

[data-testid="stMetricValue"] {
    color: #d4778b !important;
}

[data-testid="stSidebar"] {
    background-color: #e8f3e6;
}

[data-testid="stDataFrame"] {
    border: 1px solid #B8D8F5;
    border-radius: 14px;
    overflow: hidden;
}

.stButton button {
    background-color: #f4cbd3;
    color: #684b55;
    border: none;
    border-radius: 12px;
}

.stButton button:hover {
    background-color: #dcebd8;
    color: #426b50;
    border: none;
}

.stTextArea textarea {
    background-color: #EAF4FF !important;
    color: #526B85 !important;
    border: 2px solid #B8D8F5 !important;
    border-radius: 14px !important;
}

.stTextArea textarea::placeholder {
    color: #8AAAC8 !important;
}

.stTextArea textarea:focus {
    border-color: #A7C7E7 !important;
    box-shadow: 0 0 0 2px #D6E9FA !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATASET ----------------

@st.cache_data
def load_data():
    return pd.read_csv("data/tweets.csv")


df = load_data()


# ---------------- APPLY ANALYSIS ----------------

df["sentiment"] = df["tweets"].apply(analyze_sentiment)

df["emotion"] = df["tweets"].apply(detect_emotion)

df["crime_categories"] = df["tweets"].apply(
    classify_crime_multilabel
)


# ---------------- COUNTS ----------------

sentiment_order = [
    "Positive",
    "Neutral",
    "Negative"
]

emotion_order = [
    "Anger",
    "Disgust",
    "Fear",
    "Joy",
    "Sadness",
    "Surprise",
    "Neutral"
]

crime_order = [
    "phishing",
    "scam",
    "fraud",
    "hate_speech",
    "harassment",
    "misinformation",
    "none"
]

sentiment_counts = (
    df["sentiment"]
    .value_counts()
    .reindex(sentiment_order, fill_value=0)
)

emotion_counts = (
    df["emotion"]
    .value_counts()
    .reindex(emotion_order, fill_value=0)
)


# ---------------- TABLE STYLING ----------------

def style_baby_blue_table(table):

    return (
        table.style
        .set_properties(**{
            "background-color": "#EAF4FF",
            "color": "#526B85",
            "border-color": "#C5DFF5"
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#C5DFF5"),
                    ("color", "#42627D"),
                    ("border-color", "#B8D8F5")
                ]
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#EAF4FF"),
                    ("color", "#526B85"),
                    ("border-color", "#B8D8F5")
                ]
            }
        ])
    )


# ---------------- SIDEBAR NAVIGATION ----------------

st.sidebar.title("🎀 TweetLens")

st.sidebar.write("Sentiment dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Explore Dataset",
        "Sentiment Predictor",
        "Upload Dataset",
        "About"
    ]
)

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.title("🎀 TweetLens")

    st.subheader("Human Behavior Sentiment Analysis")

    st.write(
        "A dashboard for exploring sentiment "
        "and human behavior in tweets."
    )

    # Dataset overview

    st.markdown("## 💗 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Tweets", f"{len(df):,}")

    with col2:
        st.metric("Dataset Columns", len(df.columns))

    # Sentiment metrics

    st.markdown("## 🌷 Sentiment Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "💗 Positive Tweets",
            f"{sentiment_counts['Positive']:,}"
        )

    with col2:
        st.metric(
            "🌸 Neutral Tweets",
            f"{sentiment_counts['Neutral']:,}"
        )

    with col3:
        st.metric(
            "🍃 Negative Tweets",
            f"{sentiment_counts['Negative']:,}"
        )

    # Sentiment donut chart

    st.markdown("### Sentiment Distribution")

    sentiment_colors = {
        "Positive": "#F4B6C2",
        "Neutral": "#DCEBD8",
        "Negative": "#A7C7E7"
    }

    fig = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts.values,
        color=sentiment_counts.index,
        color_discrete_map=sentiment_colors,
        hole=0.55
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#526B55",
        legend_title_text="Sentiment",
        margin=dict(t=30, b=20, l=20, r=20)
    )

    fig.update_traces(
        marker=dict(
            line=dict(color="#FFF7F5", width=3)
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Sentiment bar chart

    st.markdown("### Sentiment Comparison")

    bar_fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        color=sentiment_counts.index,
        color_discrete_map=sentiment_colors,
        labels={
            "x": "Sentiment",
            "y": "Number of Tweets"
        },
        text=sentiment_counts.values
    )

    bar_fig.update_traces(
        textposition="outside",
        marker_line_color="#FFF7F5",
        marker_line_width=2
    )

    bar_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#526B55",
        showlegend=False,
        margin=dict(t=30, b=30, l=20, r=20)
    )

    st.plotly_chart(bar_fig, use_container_width=True)
    # ---------------- WORD CLOUD ----------------

    st.markdown("## ☁️ Tweet WordCloud")

    st.write(
        "Explore the most frequently used words in the tweets."
    )

    # Combine all tweet text into one string
    all_tweets = " ".join(
        df["tweets"].dropna().astype(str)
    )

    if all_tweets.strip():

        # Generate the word cloud
        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color="white",
            colormap="Pastel1",
            max_words=100
        ).generate(all_tweets)

        # Display the word cloud
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.imshow(
            wordcloud,
            interpolation="bilinear"
        )

        ax.axis("off")

        st.pyplot(fig)

        plt.close(fig)

    else:

        st.info(
            "No tweet text is available to generate a word cloud."
        )
    # Emotion bar chart

    st.markdown("## 🌸 Emotion Analysis")

    st.write(
        "Explore the frequency of different emotions "
        "identified in the tweets."
    )

    emotion_colors = {
        "Anger": "#E8A0A8",
        "Disgust": "#B9CFA8",
        "Fear": "#A7C7E7",
        "Joy": "#F4C6A6",
        "Sadness": "#B8B4DE",
        "Surprise": "#F2D58A",
        "Neutral": "#DCEBD8"
    }

    emotion_fig = px.bar(
        x=emotion_counts.index,
        y=emotion_counts.values,
        color=emotion_counts.index,
        color_discrete_map=emotion_colors,
        labels={
            "x": "Emotion",
            "y": "Number of Tweets"
        },
        text=emotion_counts.values
    )

    emotion_fig.update_traces(
        textposition="outside",
        marker_line_color="#FFF7F5",
        marker_line_width=2
    )

    emotion_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#526B55",
        showlegend=False,
        xaxis_title="Emotion",
        yaxis_title="Number of Tweets",
        margin=dict(t=30, b=30, l=20, r=20)
    )

    st.plotly_chart(
        emotion_fig,
        use_container_width=True
    )
    # ---------------- TWEET ACTIVITY TIMELINE ----------------

    st.markdown("## 📅 Tweet Activity Timeline")

    st.write(
        "Explore how tweet volume and sentiment change over time."
    )

    # Convert the time column into datetime format
    timeline_df = df.copy()

    timeline_df["time"] = pd.to_datetime(
        timeline_df["time"],
        errors="coerce"
    )

    # Remove rows where the date could not be read
    timeline_df = timeline_df.dropna(subset=["time"])

    if not timeline_df.empty:

        # Group tweets by date and sentiment
        timeline_counts = (
            timeline_df
            .groupby([
                timeline_df["time"].dt.date,
                "sentiment"
            ])
            .size()
            .reset_index(name="Tweet Count")
        )

        timeline_counts.columns = [
            "Date",
            "Sentiment",
            "Tweet Count"
        ]

        # Create timeline chart
        timeline_fig = px.line(
            timeline_counts,
            x="Date",
            y="Tweet Count",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#F4B6C2",
                "Neutral": "#DCEBD8",
                "Negative": "#A7C7E7"
            },
            markers=True,
            labels={
                "Date": "Date",
                "Tweet Count": "Number of Tweets",
                "Sentiment": "Sentiment"
            }
        )

        timeline_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#526B55",
            xaxis_title="Date",
            yaxis_title="Number of Tweets",
            legend_title_text="Sentiment",
            margin=dict(t=30, b=30, l=20, r=20)
        )

        st.plotly_chart(
            timeline_fig,
            use_container_width=True
        )

    else:

        st.info(
            "No valid dates were found in the dataset."
        )
    # Crime category heatmap

    st.markdown("## 🔎 Crime Category Heatmap")

    st.write(
        "Explore the relationship between crime-related "
        "keyword categories and sentiment labels."
    )

    # Expand tweets into one row per matched crime category.
    # Reset the index to avoid duplicate index labels after explode.

    crime_heatmap_data = (
        df.explode("crime_categories")
        .reset_index(drop=True)
    )

    # Ignore "none" because it means no category was matched.

    crime_heatmap_data = crime_heatmap_data[
        crime_heatmap_data["crime_categories"] != "none"
    ]

    # Create a matrix:
    # Rows = crime categories
    # Columns = sentiment labels
    # Values = number of tweets

    crime_sentiment_matrix = pd.crosstab(
        crime_heatmap_data["crime_categories"],
        crime_heatmap_data["sentiment"]
    )

    # Keep categories and sentiment in a consistent order.

    crime_sentiment_matrix = crime_sentiment_matrix.reindex(
        index=[
            "phishing",
            "scam",
            "fraud",
            "hate_speech",
            "harassment",
            "misinformation"
        ],
        columns=[
            "Positive",
            "Neutral",
            "Negative"
        ],
        fill_value=0
    )

    # Create the heatmap.

    crime_heatmap_fig = px.imshow(
        crime_sentiment_matrix,
        labels={
            "x": "Sentiment",
            "y": "Crime Category",
            "color": "Number of Tweets"
        },
        color_continuous_scale=[
            "#FFF7F5",
            "#F4B6C2",
            "#D4778B"
        ],
        text_auto=True,
        aspect="auto"
    )

    crime_heatmap_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#526B55",
        margin=dict(t=30, b=30, l=20, r=20)
    )

    st.plotly_chart(
        crime_heatmap_fig,
        use_container_width=True
    )

    st.caption(
        "Keyword matches are not proof that a crime occurred. "
        "A tweet may match multiple categories."
    )


# ---------------- EXPLORE DATASET ----------------

elif page == "Explore Dataset":

    st.title("🌸 Explore Dataset")

    st.write(
        "Take a closer look at the dataset used "
        "for sentiment, emotion, and crime classification."
    )

    st.markdown("### Dataset Information")

    st.write(f"**Total rows:** {len(df):,}")

    st.write(f"**Total columns:** {len(df.columns)}")

    # Sentiment filter

    st.markdown("### 💗 Filter by Sentiment")

    selected_sentiments = st.multiselect(
        "Choose sentiment(s) to display:",
        options=sentiment_order,
        default=sentiment_order
    )

    # Emotion filter

    st.markdown("### 🌸 Filter by Emotion")

    selected_emotions = st.multiselect(
        "Choose emotion(s) to display:",
        options=emotion_order,
        default=emotion_order
    )

    # Crime category filter

    st.markdown("### 🔎 Filter by Crime Category")

    selected_crimes = st.multiselect(
        "Choose crime category(s) to display:",
        options=crime_order,
        default=crime_order
    )

    # Apply all filters

    filtered_df = df[
        df["sentiment"].isin(selected_sentiments)
        & df["emotion"].isin(selected_emotions)
        & df["crime_categories"].apply(
            lambda categories: any(
                category in categories
                for category in selected_crimes
            )
        )
    ]

    st.write(f"Showing {len(filtered_df):,} tweets")

    # Dataset preview

    st.markdown("### Dataset Preview")

    if not filtered_df.empty:

        st.dataframe(
            style_baby_blue_table(filtered_df.head(100)),
            use_container_width=True
        )

        st.caption("Showing up to 100 matching tweets.")

    else:

        st.info(
            "No tweets found. Try selecting different sentiments, "
            "emotions, or crime categories."
        )

    # Sentiment counts

    st.markdown("### Sentiment Counts")

    filtered_sentiment_counts = (
        filtered_df["sentiment"]
        .value_counts()
        .reindex(sentiment_order, fill_value=0)
    )

    sentiment_table = filtered_sentiment_counts.reset_index()

    sentiment_table.columns = ["Sentiment", "Count"]

    st.dataframe(
        style_baby_blue_table(sentiment_table),
        use_container_width=True
    )

    # Emotion counts

    st.markdown("### Emotion Counts")

    filtered_emotion_counts = (
        filtered_df["emotion"]
        .value_counts()
        .reindex(emotion_order, fill_value=0)
    )

    emotion_table = filtered_emotion_counts.reset_index()

    emotion_table.columns = ["Emotion", "Count"]

    st.dataframe(
        style_baby_blue_table(emotion_table),
        use_container_width=True
    )

    # Crime category counts

    st.markdown("### Crime Category Counts")

    st.write(
        "A tweet can match more than one category, "
        "so the category counts may overlap."
    )

    if not filtered_df.empty:

        crime_counts = (
            filtered_df["crime_categories"]
            .explode()
            .value_counts()
            .reindex(crime_order, fill_value=0)
        )

        crime_table = crime_counts.reset_index()

        crime_table.columns = ["Crime Category", "Count"]

        st.dataframe(
            style_baby_blue_table(crime_table),
            use_container_width=True
        )

    else:

        st.info("No crime category counts to display.")


# ---------------- SENTIMENT PREDICTOR ----------------

elif page == "Sentiment Predictor":

    st.title("💌 Sentiment Predictor")

    st.write(
        "Enter a sentence or tweet below "
        "to predict its sentiment and emotion."
    )

    user_text = st.text_area(
        "Your text",
        placeholder="Type something here...",
        height=150
    )

    if st.button("Analyze Text 🎀"):

        if user_text.strip():

            result = analyze_sentiment(user_text)

            emotion_result = detect_emotion(user_text)

            st.markdown("### Results")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("#### Sentiment")

                if result == "Positive":
                    st.success(f"💗 {result}")

                elif result == "Negative":
                    st.error(f"🍃 {result}")

                else:
                    st.info(f"🌸 {result}")

            with col2:

                st.markdown("#### Emotion")

                st.info(f"🎀 {emotion_result}")

            st.caption(
                "Emotion detection is keyword-based and may not "
                "understand context or sarcasm."
            )

        else:

            st.warning("Please enter some text first!")
# ---------------- UPLOAD DATASET ----------------

elif page == "Upload Dataset":

    st.title("📂 Upload & Export Dataset")

    st.write(
        "Upload a CSV file containing tweets "
        "to analyze their sentiment, emotion, "
        "and crime categories."
    )

    # Upload CSV file
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        # Read uploaded CSV
        uploaded_df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.markdown("### Dataset Preview")

        st.dataframe(
            uploaded_df.head(),
            use_container_width=True
        )

        # Check if tweet column exists
        if "tweets" in uploaded_df.columns:

            st.info(
                "Your dataset contains a 'tweets' column."
            )

            # Analyze sentiment
            uploaded_df["sentiment"] = (
                uploaded_df["tweets"]
                .astype(str)
                .apply(analyze_sentiment)
            )

            # Analyze emotion
            uploaded_df["emotion"] = (
                uploaded_df["tweets"]
                .astype(str)
                .apply(detect_emotion)
            )

            # Analyze crime categories
            uploaded_df["crime_categories"] = (
                uploaded_df["tweets"]
                .astype(str)
                .apply(classify_crime_multilabel)
            )

            st.markdown("### Analyzed Dataset")

            st.dataframe(
                uploaded_df,
                use_container_width=True
            )

            # Export analyzed dataset
            csv_data = uploaded_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Analyzed CSV",
                data=csv_data,
                file_name="tweetlens_analyzed_dataset.csv",
                mime="text/csv"
            )

        else:

            st.warning(
                "The uploaded CSV must contain a column named 'tweets'."
            )

# ---------------- ABOUT ----------------

elif page == "About":

    st.title("🌷 About TweetLens")

    st.write(
        "TweetLens is a Human Behavior Sentiment Analysis "
        "project designed to explore sentiment, emotions, "
        "and possible crime-related keywords in tweets."
    )

    st.markdown("### What does it do?")

    st.write("""
    - Analyzes tweets using VADER sentiment analysis.
    - Classifies text as Positive, Negative, or Neutral.
    - Identifies possible emotions using keyword matching.
    - Identifies possible crime-related categories using keywords.
    - Displays sentiment distribution through a dashboard.
    - Displays emotion frequency through a bar chart.
    - Allows users to explore and filter the dataset.
    - Includes a simple sentiment and emotion prediction feature.
    """)

    st.markdown("### Technologies Used")

    st.write("""
    - Python
    - Streamlit
    - Pandas
    - Plotly
    - VADER Sentiment
    - Regular Expressions
    """)

    st.markdown("### Emotion Detection Note")

    st.write(
        "The emotion detector uses a rule-based keyword lexicon "
        "and returns the emotion with the highest keyword count. "
        "It is not a trained machine-learning model."
    )

    st.markdown("### Crime Classification Note")

    st.write(
        "Crime classification uses keyword matching to identify "
        "possible categories. These matches do not confirm that "
        "a crime occurred and may misclassify tweets depending "
        "on context."
    )
