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
import openai
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from heapq import nlargest
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import sentencepiece
from transformers import BertTokenizer, BertForMaskedLM
from transformers import T5ForConditionalGeneration, T5Tokenizer
import random
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
    st.sidebar.title("Selección de Columnas")
    selected_columns = st.sidebar.multiselect("Selecciona las columnas para el análisis de NLP", df.columns)

    st.sidebar.title("Análisis de NLP")
    analysis_option = st.sidebar.selectbox("Selecciona una opción de análisis de NLP", ["Ninguno", "Análisis de Sentimiento", 
                                                                          "Emociones", "Nube de Palabras", "Frecuencia", "Resumen de Texto"])

    return selected_columns, analysis_option

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

def generate_wordcloud(df, selected_columns):
    text = ""
    for column in selected_columns:
        text += " ".join(df[column].astype(str)) + " "

    words = word_tokenize(text, language='spanish')

    stop_words = set(stopwords.words("spanish"))
    words = [word for word in words if word.lower() not in stop_words and word.isalpha()]

    processed_text = " ".join(words)

    wordcloud = WordCloud(background_color='white', width=800, height=400, colormap='gist_rainbow', color_func=lambda *args, **kwargs: "#800040").generate(processed_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt.gcf())

    wordcloud_image = wordcloud.to_image()

    img_bytes = io.BytesIO()
    wordcloud_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    st.download_button("Descargar Nube de Palabras", data=img_bytes, file_name='nube_de_palabras.png', mime='image/png')

def generate_frequency_analysis(df, selected_columns):
    text = ""
    for column in selected_columns:
        text += " ".join(df[column].astype(str)) + " "

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

def perform_sentiment_analysis(df, text_column):
    sid = SentimentIntensityAnalyzer()
    df['Puntaje de Sentimiento'] = df[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
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

def assign_emotions(df, text_column):
    emociones = {
    'Felicidad': ['feliz', 'alegría', 'emocionado', 'encantado', 'extasiado', 'jubiloso', 'contento', 'eufórico', 'radiante', 'sonriente', 'gozoso', 'regocijo', 'exultante', 'euforia', 'divertido', 'pleno', 'venturoso', 'afortunado'],
    'Tristeza': ['triste', 'deprimido', 'melancólico', 'desconsolado', 'apenado', 'abatido', 'lloroso', 'desesperado', 'melancolía', 'penoso', 'lamentable', 'agonía', 'desgarrador', 'desolado', 'amargado', 'sombrío', 'duro', 'sin esperanza'],
    'Enojo': ['enojado', 'frustrado', 'indignado', 'irritado', 'furioso', 'rabioso', 'exasperado', 'agresivo', 'hostil', 'iracundo', 'ardiente', 'belicoso', 'vengativo', 'furibundo', 'colérico', 'enfadado', 'resentido', 'resentimiento'],
    'Celos': ['celoso', 'envidioso', 'codicioso', 'inseguro', 'competitivo', 'resentido', 'desconfiado', 'obsesivo', 'posesivo', 'verde de envidia', 'desear lo que otros tienen', 'lamentar el éxito ajeno', 'recelo', 'despecho', 'invidiar', 'celotipia', 'avaricia'],
    'Amor': ['amor', 'adorar', 'afecto', 'romántico', 'apasionado', 'ternura', 'cariño', 'devoción', 'comprensión', 'afinidad', 'pasión ardiente', 'flechazo', 'idilio', 'encantamiento', 'afecto profundo', 'amor eterno', 'seducción', 'cortejar'],
    'Miedo': ['miedo', 'ansiedad', 'preocupado', 'aterrorizado', 'temeroso', 'nervioso', 'asustado', 'pánico', 'horror', 'espanto', 'terrorífico', 'inseguridad', 'aprehensión', 'miedoso', 'apatía', 'temor paralizante', 'pavor', 'miedo irracional'],
    'Sorpresa': ['sorpresa', 'asombrado', 'impactado', 'maravillado', 'atónito', 'estupefacto', 'sobrecogido', 'asombro', 'desconcertado', 'asombroso', 'deslumbrante', 'estupefacción', 'deslumbrado', 'estático', 'perplejidad', 'deslumbramiento', 'maravilloso', 'incredulidad'],
    'Asco': ['asco', 'repulsión', 'náuseas', 'repugnancia', 'aversión', 'desagrado', 'nauseabundo', 'indignante', 'abominable', 'asqueroso', 'repelente', 'detestable', 'vómito', 'antipatía', 'desprecio', 'repulsivo', 'asquerosidad'],
    'Confusión': ['confundido', 'perplejo', 'desconcertado', 'desorientado', 'perdido', 'atolondrado', 'despistado', 'incertidumbre', 'ambigüedad', 'perplejidad', 'desconcertante', 'desordenado', 'lío', 'caos', 'confusión mental', 'desorientación', 'desubicado', 'desconcierto'],
    'Emoción': ['emoción', 'emocionado', 'ansioso', 'sentimiento', 'pasión', 'excitación', 'inquietud', 'entusiasmo', 'agitado', 'sentir', 'estímulo', 'experiencia', 'reacción', 'sensación', 'intensidad', 'sensibilidad', 'vivacidad', 'efervescencia'],
    'Esperanza': ['esperanza', 'optimista', 'expectativa', 'esperanzado', 'ilusión', 'positivo', 'confiado', 'esperar', 'anhelo', 'fe', 'anticipación', 'esperanzador', 'promesa', 'aspiración', 'metas', 'visión', 'optimismo'],
    'Orgullo': ['orgullo', 'orgulloso', 'realizado', 'satisfacción', 'egocentrismo', 'vanidad', 'arrogancia', 'triunfante', 'dignidad', 'soberbia', 'autoestima', 'logro', 'gloria', 'respeto propio', 'autoafirmación', 'alabanza', 'autoconfianza'],
    'Culpabilidad': ['culpa', 'arrepentido', 'remordimiento', 'culpable', 'penitencia', 'lamentar', 'autoacusación', 'vergüenza', 'autocondena', 'autoreproche', 'culpabilidad moral', 'culpabilidad emocional', 'autoexigencia', 'autopunición', 'inocencia perdida', 'perdón', 'autocrítica'],
    'Vergüenza': ['vergüenza', 'avergonzado', 'humillado', 'sonrojado', 'inhibición', 'desprestigio', 'desgracia', 'timidez', 'autoconciencia', 'bochorno', 'mortificación', 'incomodidad', 'arrepentimiento', 'sentimiento de culpa', 'sonrojo', 'disgusto', 'vergonzoso'],
    'Alivio': ['alivio', 'consolado', 'tranquilizado', 'sosiego', 'respiro', 'relajación', 'descanso', 'paz', 'desahogo', 'liberación', 'relevo', 'reconfortante', 'relief', 'relación', 'esperanza cumplida', 'calma', 'seguridad'],
    'Curiosidad': ['curiosidad', 'inquisitivo', 'exploración', 'interés', 'inquietud', 'deseo de saber', 'preguntas', 'investigación', 'avidez', 'descubrimiento', 'indagación', 'curiosear', 'intriga', 'novedad', 'indagar', 'sed de conocimiento', 'saber más'],
    'Nostalgia': ['nostalgia', 'sentimental', 'recuerdo', 'añoranza', 'melancolía', 'reminiscencia', 'tristeza por el pasado', 'retroceso', 'recuerdos pasados', 'recordar', 'memoria', 'melancólico', 'suspiro', 'recordar con cariño', 'volver atrás', 'pasado'],
    }

    def get_emotion(text):
        emotions_detected = {emotion: 0 for emotion in emociones}
        text = text.lower()
        words = text.split()

        for word in words:
            for emotion, keywords in emociones.items():
                if word in keywords:
                    emotions_detected[emotion] += 1

        max_emotion = max(emotions_detected, key=emotions_detected.get)
        return max_emotion

    df['Emoción Dominante'] = df[text_column].apply(get_emotion)
    return df

def plot_emotion_distribution(df):
    st.subheader("Análisis de Emociones")
    emotion_counts = df['Emoción Dominante'].value_counts()
    st.write(emotion_counts)

    fig, ax = plt.subplots()
    emotion_counts.plot(kind='bar', color='#800040', ax=ax)
    ax.set_xlabel("Emoción")
    ax.set_ylabel("Conteo")
    ax.set_title("Distribución de Emociones")
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
    selected_columns, analysis_option = render_sidebar(df)

    if selected_columns:
        st.header("Datos Seleccionados")
        st.dataframe(df[selected_columns])

    if analysis_option == "Análisis de Sentimiento":
        df = perform_sentiment_analysis(df, selected_columns[0])
        plot_sentiment_scores(df)

    elif analysis_option == "Emociones":
        df = assign_emotions(df, selected_columns[0])
        plot_emotion_distribution(df)

    elif analysis_option == "Nube de Palabras":
        generate_wordcloud(df, selected_columns)

    elif analysis_option == "Frecuencia":
        generate_frequency_analysis(df, selected_columns)

    elif analysis_option == "Resumen de Texto":
        perform_text_summarization(df, selected_columns[0])
