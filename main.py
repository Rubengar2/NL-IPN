import streamlit as st
import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
import seaborn as sns
import numpy as np
from PIL import Image
from collections import Counter
import io
import nltk
import os
import string
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from heapq import nlargest
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords(text):
    stop_words = set(stopwords.words("spanish"))
    punctuation = set(string.punctuation)
    words = nltk.word_tokenize(text, language='spanish')  # Tokenización en español
    words = [word for word in words if word.lower() not in stop_words and word not in punctuation]
    return " ".join(words)

def render_sidebar(df):
    st.sidebar.title("Column Selection")
    selected_columns = st.sidebar.multiselect("Select columns for NLP analysis", df.columns)

    st.sidebar.title("NLP Analysis")
    analysis_option = st.sidebar.selectbox("Select NLP analysis option", ["None", "Sentiment Analysis", 
                                                                          "Emotions", "Wordcloud", "Frequency","Text Summarization"])

    return selected_columns, analysis_option

def generate_summary(text, num_sentences=3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Remove stopwords from sentences
    stop_words = set(stopwords.words('english'))
    word_frequencies = {}
    for word in word_tokenize(text):
        if word.lower() not in stop_words:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Calculate sentence scores based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequencies[word]
                else:
                    sentence_scores[sentence] += word_frequencies[word]

    # Get the top N sentences with highest scores
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)

    return summary

def perform_text_summarization(df, text_column):
    st.subheader("Text Summarization")

    selected_text = " ".join(df[text_column].astype(str))

    if st.button("Generate Summary"):
        summary = generate_summary(selected_text)
        st.write(summary)

def generate_wordcloud(df, selected_columns):
    text = ""
    for column in selected_columns:
        text += " ".join(df[column].astype(str)) + " "

    # Tokenización en español
    words = word_tokenize(text, language='spanish')

    # Eliminar stopwords en español y puntuación
    stop_words = set(stopwords.words("spanish"))
    words = [word for word in words if word.lower() not in stop_words and word.isalpha()]

    # Unir las palabras procesadas
    processed_text = " ".join(words)

    # Generar la Wordcloud usando una máscara e imagen predeterminada
    wordcloud = WordCloud(background_color='white', width=800, height=400, colormap='gist_rainbow', color_func=lambda *args, **kwargs: "#800040").generate(processed_text)

    # Mostrar la imagen de la Wordcloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt.gcf())

    # Convertir la Wordcloud a una imagen
    wordcloud_image = wordcloud.to_image()

    # Crear un flujo en memoria para almacenar los datos de la imagen
    img_bytes = io.BytesIO()
    wordcloud_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Proporcionar un botón de descarga para la imagen de la Wordcloud
    st.download_button("Download Wordcloud", data=img_bytes, file_name='wordcloud.png', mime='image/png')


def generate_frequency_analysis(df, selected_columns):
    text = ""
    for column in selected_columns:
        text += " ".join(df[column].astype(str)) + " "

    text = remove_stopwords(text)

    word_list = text.split()
    word_counter = Counter(word_list)
    word_counter = dict(word_counter.most_common(10))  # Select top 10 most common words

    freq_df = pd.DataFrame.from_dict(word_counter, orient='index', columns=['Frequency'])
    freq_df.index.name = 'Word'
    freq_df.reset_index(inplace=True)

    st.dataframe(freq_df)

    # Plot top 10 most repeated words
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(freq_df['Word'], freq_df['Frequency'], color='#800040')
    ax.set_xlabel('Word')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 10 Most Repeated Words')
    ax.set_xticklabels(freq_df['Word'], rotation=45)

    # Add frequency labels inside the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval),
                ha='center', va='bottom', color='black', fontsize=8, weight='bold')

    st.pyplot(fig)

def perform_sentiment_analysis(df, text_column):
    sid = SentimentIntensityAnalyzer()
    df['Sentiment Score'] = df[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
    return df


def plot_sentiment_scores(df):
    st.subheader("Sentiment Analysis")
    
    sentiment_scores = df['Sentiment Score']
    
    # Plot the distribution of sentiment scores
    fig, ax = plt.subplots()
    sns.histplot(sentiment_scores, kde=True, color='#800040', ax=ax)
    ax.set(xlabel="Sentiment Score", ylabel="Count")
    ax.set_title("Distribution of Sentiment Scores")
    st.pyplot(fig)

    # Calculate the most representative sentiments
    top_sentiments = sentiment_scores.value_counts().nlargest(5)
    st.subheader("Most Representative Sentiments")
    st.write(top_sentiments)

def assign_emotions(df, text_column):
    # Define emotions and associated keywords
    emotions = {
    'Happiness': ['happy', 'joy', 'excited', 'delighted'],
    'Sadness': ['sad', 'depressed', 'gloomy', 'heartbroken'],
    'Anger': ['angry', 'frustrated', 'outraged', 'irritated'],
    'Jealousy': ['jealous', 'envious', 'covetous'],
    'Love': ['love', 'adore', 'affection', 'romantic'],
    'Fear': ['fear', 'anxiety', 'worried', 'terrified'],
    'Surprise': ['surprise', 'astonished', 'amazed', 'shocked'],
    'Disgust': ['disgust', 'repulsed', 'nauseated'],
    'Confusion': ['confused', 'puzzled', 'perplexed'],
    'Excitement': ['excitement', 'thrilled', 'eager'],
    'Hope': ['hope', 'optimistic', 'expectation'],
    'Pride': ['pride', 'proud', 'accomplished'],
    'Guilt': ['guilt', 'regretful', 'remorseful'],
    'Shame': ['shame', 'ashamed', 'humiliated'],
    'Relief': ['relief', 'comforted', 'reassured'],
    'Curiosity': ['curiosity', 'inquisitive', 'exploration'],
    'Nostalgia': ['nostalgia', 'sentimental', 'remembrance'],
    'Contempt': ['contempt', 'disdain', 'disrespectful'],
    'Amusement': ['amusement', 'entertained', 'tickled'],
    'Gratitude': ['gratitude', 'thankful', 'appreciative']
    }

    # Initialize emotion counter
    emotion_counter = Counter()

    # Iterate over each comment
    for comment in df[text_column]:
        comment = comment.lower()

        # Check for keyword matches with emotions
        for emotion, keywords in emotions.items():
            for keyword in keywords:
                if keyword in comment:
                    emotion_counter[emotion] += 1
                    break

    # Get the top 5 most registered emotions
    top_emotions = emotion_counter.most_common(5)

    return top_emotions

def perform_emotion_analysis(df, text_column):
    sid = SentimentIntensityAnalyzer()
    df['Sentiment Score'] = df[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])

    # Assign emotions to comments and get top 5 emotions
    top_emotions = assign_emotions(df, text_column)

    # Plot count of top emotions
    fig, ax = plt.subplots()
    x, y = zip(*top_emotions)
    ax.bar(x, y, color='#800040')
    ax.set(xlabel='Emotion', ylabel='Count')
    ax.set_title('Count of Emotions in Comments')
    ax.tick_params(axis='x', rotation=45)

    # Add count labels
    for i, count in enumerate(y):
        ax.text(i, count, str(count), ha='center', va='bottom', color='black')

    # Print the full table of emotion counts
    emotion_table = pd.DataFrame(top_emotions, columns=['Emotion', 'Count'])
    total_count = emotion_table['Count'].sum()
    emotion_table.loc[len(emotion_table)] = ['Total', total_count]
    st.dataframe(emotion_table.style.set_caption('Emotion Count'))

    # Display the plot
    st.pyplot(fig)

    return df
#inicio de la funcion principal
def main():
    st.title("NLP Analysis App - Main v.0.0.2")
    #Cambio en la version titulo    

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
                    generate_wordcloud(df, selected_columns) 

                elif analysis_option == "Frequency":
                    generate_frequency_analysis(df, selected_columns)

                elif analysis_option == "Text Summarization":
                    text_column = st.selectbox("Select the column containing the text", selected_columns)
                    perform_text_summarization(selected_df, text_column)

                elif analysis_option == "Emotions":
                    text_column = st.selectbox("Select the column containing the text", selected_columns)
                    perform_emotion_analysis(selected_df, text_column)
    
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
