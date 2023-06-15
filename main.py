import streamlit as st
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
import numpy as np
from PIL import Image
import io
import nltk
import os
import string
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    punctuation = set(string.punctuation)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words and word not in punctuation]
    return " ".join(words)

def render_sidebar(df):
    st.sidebar.title("Column Selection")
    selected_columns = st.sidebar.multiselect("Select columns for NLP analysis", df.columns)

    st.sidebar.title("NLP Analysis")
    analysis_option = st.sidebar.selectbox("Select NLP analysis option", ["None", "Sentiment Analysis", "Wordcloud", "Frequency"])

    return selected_columns, analysis_option

def generate_wordcloud(df, selected_columns, mask_image_path):
    text = ""
    for column in selected_columns:
        text += " ".join(df[column].astype(str)) + " "

    mask_image = np.array(Image.open(mask_image_path))

    wordcloud = WordCloud(background_color='white', mask=mask_image).generate(text)

    # Apply colors from the mask image to the Wordcloud
    image_colors = ImageColorGenerator(mask_image)
    colored_wordcloud = wordcloud.recolor(color_func=image_colors)

    # Convert the Wordcloud to an image
    image = colored_wordcloud.to_image()

    # Display the Wordcloud image
    st.image(image)

    # Create an in-memory stream for storing the image data
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Prompt the user to download the Wordcloud image
    st.download_button("Download Wordcloud", data=img_bytes, file_name='wordcloud.png', mime='image/png')

def generate_frequency_analysis(df, selected_columns):
    text = ""
    for column in selected_columns:
        text += " ".join(df[column].astype(str)) + " "

    text = remove_stopwords(text)

    word_list = text.split()
    word_counter = Counter(word_list)
    word_counter = dict(word_counter.most_common())

    freq_df = pd.DataFrame.from_dict(word_counter, orient='index', columns=['Frequency'])
    freq_df.index.name = 'Word'
    freq_df.reset_index(inplace=True)

    st.dataframe(freq_df)

def perform_sentiment_analysis(df, text_column):
    sid = SentimentIntensityAnalyzer()
    df['Sentiment Score'] = df[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
    return df

def plot_sentiment_scores(df):
    st.subheader("Sentiment Scores")
    st.dataframe(df)

    st.subheader("Sentiment Score Distribution")
    st.bar_chart(df['Sentiment Score'])

def main():
    st.title("NLP Analysis App - Main")

    uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt", "xls", "xlsx"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]

        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == "txt":
            df = pd.read_csv(uploaded_file, delimiter="\t")
        elif file_extension in ["xls", "xlsx"]:
            df = pd.read_excel(uploaded_file)

        selected_columns, analysis_option = render_sidebar(df)

        if analysis_option != "None":
            st.dataframe(df.head())

            if selected_columns:
                selected_df = df[selected_columns]

                if analysis_option == "Sentiment Analysis":
                    text_column = st.selectbox("Select the column containing the text", selected_columns)
                    sentiment_df = perform_sentiment_analysis(selected_df, text_column)
                    plot_sentiment_scores(sentiment_df)

                elif analysis_option == "Wordcloud":
                    mask_image_path = st.file_uploader("Upload a mask image", type=["png"])

                    if mask_image_path is not None:
                        generate_wordcloud(df, selected_columns, mask_image_path)

                elif analysis_option == "Frequency":
                    generate_frequency_analysis(df, selected_columns)
    
    if st.button("Home"):
        st.experimental_set_query_params(page="menu")

def presentation_menu():
    st.markdown(
        """
        <style>
        .center {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 0vh;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    columns = st.columns([1, 2, 1])
    for _ in range(2):
        columns[_].write("")

    with columns[1]:
        st.markdown('<div class="center">', unsafe_allow_html=True)
        st.title("Welcome to NLP Analysis App - Presentation Menu")
        st.subheader("Click the button below to start the analysis")
        
        # Use a state variable to track the button click
        button_clicked = st.button("Start")
        if button_clicked:
            st.experimental_set_query_params(page="main")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", ["menu"])[0]

    if page == "menu":
        presentation_menu()
    elif page == "main":
        main()
