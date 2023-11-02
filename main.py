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
import torch
import string
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "finiteautomata/beto-emotion-analysis"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords(text):
    stop_words = set(stopwords.words("spanish"))
    punctuation = set(string.punctuation)
    words = nltk.word_tokenize(text, language='spanish')
    words = [word for word in words if word.lower() not in stop_words and word not in punctuation]
    return " ".join(words)

def render_sidebar(df):
    st.sidebar.title("Selección de Columnas")
    selected_columns = st.sidebar.multiselect("Selecciona las columnas para el análisis de NLP", df.columns)

    st.sidebar.title("Análisis de NLP")
    analysis_option = st.sidebar.selectbox("Selecciona una opción de análisis de NLP", ["Ninguno", "Análisis de Sentimiento",
                                                                          "Emociones", "Nube de Palabras", "Frecuencia", "Resumen de Texto"])

    sentiment_label = st.sidebar.selectbox("Selecciona el tipo de comentarios", ["Todos", "Positivos", "Negativos"])

    return selected_columns, analysis_option, sentiment_label

def generate_summary_with_t5(text, max_length=200):
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    inputs = tokenizer.encode("resumir: " + text, return_tensors="pt", max_length=1024, truncation=True)

    summary_ids = model.generate(inputs, max_length=max_length, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def perform_text_summarization(df, text_column):
    st.subheader("Resumen de Texto")

    selected_comments = df[text_column].astype(str).tolist()

    if st.button("Generar Resumen"):
        combined_text = " ".join(selected_comments)
        summary = generate_summary_with_t5(combined_text)
        st.write(summary)

def generate_wordcloud(df, selected_columns, sentiment_label):
    text = ""
    for column in selected_columns:
        text += " ".join(df[column].astype(str)) + " "

    sid = SentimentIntensityAnalyzer()
    df['Puntaje de Sentimiento'] = df[selected_columns[0]].apply(lambda x: sid.polarity_scores(x)['compound'])

    if sentiment_label == "Positivos":
        df = df[df['Puntaje de Sentimiento'] > 0]
    elif sentiment_label == "Negativos":
        df = df[df['Puntaje de Sentimiento'] < 0]

    words = word_tokenize(" ".join(df[selected_columns[0]].astype(str)), language='spanish')

    stop_words = set(stopwords.words("spanish"))
    words = [word for word in words if word.lower() not in stop_words and word.isalpha()]

    processed_text = " ".join(words)

    wordcloud = WordCloud(background_color='white', width=800, height=400, colormap='gist_rainbow',
                          color_func=lambda *args, **kwargs: "#800040").generate(processed_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt.gcf())

    wordcloud_image = wordcloud.to_image()

    img_bytes = io.BytesIO()
    wordcloud_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    st.download_button("Descargar Nube de Palabras", data=img_bytes, file_name='nube_de_palabras.png', mime='image/png')

def generate_frequency_analysis(df, selected_columns, sentiment_label):
    text = ""
    for column in selected_columns:
        text += " ".join(df[column].astype(str)) + " "

    # Calcular los puntajes de sentimiento para todos los comentarios
    sid = SentimentIntensityAnalyzer()
    df['Puntaje de Sentimiento'] = df[selected_columns[0]].apply(lambda x: sid.polarity_scores(x)['compound'])

    if sentiment_label == "Positivos":
        df = df[df['Puntaje de Sentimiento'] > 0]
    elif sentiment_label == "Negativos":
        df = df[df['Puntaje de Sentimiento'] < 0]

    # Realizar el análisis de frecuencia
    text = " ".join(df[selected_columns[0]].astype(str))
    text = remove_stopwords(text)

    word_list = text.split()
    word_counter = Counter(word_list)
    word_counter = dict(word_counter.most_common(10))

    freq_df = pd.DataFrame.from_dict(word_counter, orient='index', columns=['Frecuencia'])
    freq_df.index.name = 'Palabra'
    freq_df.reset_index(inplace=True)

    st.dataframe(freq_df)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(freq_df['Palabra'], freq_df['Frecuencia'], color='#800040')
    ax.set_xlabel('Palabra')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Top 10 Palabras Más Repetidas')
    ax.set_xticklabels(freq_df['Palabra'], rotation=45)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, int(yval),
                ha='center', va='bottom', color='black', fontsize=8, weight='bold')

    st.pyplot(fig)

def perform_sentiment_analysis(df, text_column, sentiment_label):
    sid = SentimentIntensityAnalyzer()
    df['Puntaje de Sentimiento'] = df[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])

    if sentiment_label == "Positivos":
        df = df[df['Puntaje de Sentimiento'] > 0]
    elif sentiment_label == "Negativos":
        df = df[df['Puntaje de Sentimiento'] < 0]

    st.subheader("Comentarios con Polaridad " + sentiment_label)
    st.dataframe(df)

    return df

def plot_sentiment_scores(df):
    st.subheader("Análisis de Sentimiento")
    
    sentiment_scores = df['Puntaje de Sentimiento']
    
    fig, ax = plt.subplots()
    sns.histplot(sentiment_scores, kde=True, color='#800040', ax=ax)
    ax.set(xlabel="Puntaje de Sentimiento", ylabel="Conteo")
    ax.set_title("Distribución de los Puntajes de Sentimiento")
    st.pyplot(fig)

    top_sentiments = sentiment_scores.value_counts().nlargest(5)
    st.subheader("Sentimientos Más Representativos")
    st.write(top_sentiments)

def perform_emotion_analysis_beto(comments, sentiment_label):
    emotions = []
    for comment in comments:
        inputs = tokenizer(comment, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
        emotion_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        
        # Asegúrate de que la etiqueta generada esté dentro del rango válido
        if predicted_label < 0:
            predicted_label = 0
        elif predicted_label >= len(emotion_labels):
            predicted_label = len(emotion_labels) - 1
        
        emotion = emotion_labels[predicted_label]
        emotions.append(emotion)
    
    # Filtra los comentarios en función del sentimiento
    if sentiment_label == "Positivos":
        emotions = [emotion for emotion in emotions if emotion in ["joy", "surprise"]]
    elif sentiment_label == "Negativos":
        emotions = [emotion for emotion in emotions if emotion in ["anger", "disgust", "fear", "sadness"]]
    
    return emotions

def plot_emotion_distribution(df, selected_columns, sentiment_label):
    st.subheader("Análisis de Emociones")

    # Filtra los comentarios en función del sentimiento
    emotions = perform_emotion_analysis_beto(df[selected_columns[0]], sentiment_label)

    # Mapeo de emociones en inglés a español
    emotion_mapping = {
        "anger": "enojo",
        "disgust": "asco",
        "fear": "miedo",
        "joy": "alegría",
        "sadness": "tristeza",
        "surprise": "sorpresa"
    }

    emotions_in_spanish = [emotion_mapping[emotion] for emotion in emotions]

    st.write(emotions_in_spanish)

    # Create a bar plot
    fig, ax = plt.subplots()
    unique_emotions = list(set(emotions_in_spanish))  # Get unique emotions
    emotion_counts = [emotions_in_spanish.count(emotion) for emotion in unique_emotions]  # Count occurrences

    ax.bar(unique_emotions, emotion_counts, color='#800040')
    ax.set_xlabel("Emoción")
    ax.set_ylabel("Conteo")
    ax.set_title("Distribución de Emociones")
    plt.xticks(rotation=45)

    # Agregar etiquetas con números de conteo en las barras
    for i, v in enumerate(emotion_counts):
        ax.text(i, v, str(v), color='black', ha='center', va='bottom', fontsize=8, weight='bold')

    st.pyplot(fig)



st.title("Análisis de Texto en Español")

st.header("Carga de Datos")
uploaded_file = st.file_uploader("Carga tu archivo de datos (CSV o Excel)", type=["csv", "xlsx"])
if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1]
    if file_ext == ".csv":
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    elif file_ext == ".xlsx":
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    else:
        st.error("El formato de archivo no es compatible. Carga un archivo CSV o Excel.")
else:
    st.info("Carga un archivo de datos para continuar.")

if 'df' in locals():
    selected_columns, analysis_option, sentiment_label = render_sidebar(df)

    if selected_columns:
        st.header("Datos Seleccionados")
        st.dataframe(df[selected_columns])

    if analysis_option == "Análisis de Sentimiento":
        df = perform_sentiment_analysis(df, selected_columns[0], sentiment_label)
        plot_sentiment_scores(df)

    if analysis_option == "Emociones":
        emotions = perform_emotion_analysis_beto(df[selected_columns[0]], sentiment_label)
        plot_emotion_distribution(df, selected_columns, sentiment_label)
        df['Emoción Dominante'] = emotions
        emotion_counts = df['Emoción Dominante'].value_counts()
        st.write(emotion_counts)

    elif analysis_option == "Nube de Palabras":
        generate_wordcloud(df, selected_columns, sentiment_label)

    elif analysis_option == "Frecuencia":
        generate_frequency_analysis(df, selected_columns, sentiment_label)

    elif analysis_option == "Resumen de Texto":
        perform_text_summarization(df, selected_columns[0])
        
        
