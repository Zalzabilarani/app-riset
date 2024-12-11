import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import spacy
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet', quiet=False)

nlp = spacy.load("en_core_web_sm")

# Load your pre-trained models and vectorizer
kmeans = joblib.load('kmeans_model.pkl')
svm_classifier = joblib.load('svm_model.pkl')
smote = joblib.load('smote_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

if 'df' not in st.session_state:
    st.session_state['df'] = None  # Awalnya kosong

# Streamlit app
st.set_page_config(page_title="Aplikasi Klasifikasi dan Sentimen untuk Menganalisis Teks Ulasan", page_icon="📝", layout="wide")
st.title("Aplikasi Klasifikasi dan Sentimen untuk Menganalisis Teks Ulasan")

# Navigation menu
pages = {
    "Home": 1,
    "Preprocessing": 2,
    "Word Cloud": 3,
    "Feature Extraction": 4,
    "SMOTE": 5,
    "Model Performance": 6,
    "Test": 7,
}

selected_page = st.sidebar.radio("Select a page", options=list(pages.keys()))

# Mode selection (dark or light)
mode = st.sidebar.radio("Select Theme", options=["Light Mode", "Dark Mode"], key="theme_mode")

# Define light and dark themes using CSS
if mode == "Dark Mode":
    st.markdown(
        """
        <style>
        body {
            background-color: #121212;
            color: white;
        }
        .streamlit-expanderHeader {
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #2e2e2e;
        }
        .stButton>button {
            background-color: #6200ea;
            color: white;
        }
        .stSelectbox>div>input {
            background-color: #2e2e2e;
            color: white;
        }
        .stRadio>div>label {
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
else:
    st.markdown(
        """
        <style>
        body {
            background-color: white;
            color: black;
        }
        .stButton>button {
            background-color: #6200ea;
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #f4f4f4;
        }
        .stSelectbox>div>input {
            background-color: white;
            color: black;
        }
        .stRadio>div>label {
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)

if pages[selected_page] == 1:
    st.write("""
        Aplikasi ini menunjukkan analisis sentimen ulasan Twitter mengenai **Karapan Sapi**, 
        acara balap banteng tradisional dari Madura, Indonesia. Proyek ini menggunakan **K-Means Clustering**
        untuk mengidentifikasi berbagai kelompok sentimen dan **Support Vector Machine (SVM)** untuk mengklasifikasikan sentimen
        sebagai positif dan negatif.
        
        Tujuan: Sasaran analisis ini adalah untuk mengeksplorasi opini publik dan sentimen yang diungkapkan di Twitter tentang Karapan Sapi, mengungkap bagaimana orang memandang acara tersebut dan makna budaya di baliknya.

        **Pendekatan**:
        1. **Pengumpulan Data**: Ulasan Twitter terkait Karapan Sapi dikumpulkan melalui API atau web scraping.
        2. **Praproses**: Data teks diproses terlebih dahulu menggunakan teknik seperti tokenisasi, penghapusan stopword, dan lemmatisasi.
        3. **Pengelompokan dengan K-Means**: Kami menerapkan pengelompokan K-Means untuk mengungkap pola dalam sentimen dan mengelompokkan opini yang serupa.
        4. **Klasifikasi Sentimen dengan SVM**: Support Vector Machine (SVM) digunakan untuk mengklasifikasikan sentimen ulasan sebagai positif, negatif, atau netral.
        5. **Evaluasi**: Hasil dievaluasi menggunakan metrik kinerja seperti akurasi, presisi, recall, dan skor F1.
    """)
    
    sample_data = {
        'Tweet': [
            "abigailimuriaa  perawat kaki kuda  eo karapan sapi",
            "habis nonton film aanwell prestasinya lomba melamun kab bogor sama lomba karapan sapi",
            "ya yg kegiatan laki laki gitu tawuran ngebengkel karapan sapi futsal panjat pinang",
            "panglima tni lestarikan budaya melalui karapan sapi perang bintang",
            "laksamana budayawan gelar pesta budaya lomba karapan sapi panglima tni cup."
        ],
        'Sentiment': ['Positif', 'Positif', 'Positif', 'Negatif', 'Negatif']
    }

    df_sample = pd.DataFrame(sample_data)

    st.write("### Contoh Data dari Penelitian:")
    st.dataframe(df_sample)

    st.write("""
        Data yang digunakan dalam proyek ini adalah kumpulan ulasan Twitter 
        tentang Karapan Sapi, termasuk sentimen positif dan negatif. Berikut 
        ini adalah pratinjau beberapa ulasan dan klasifikasi sentimen terkait.

        Alur kerja analisis sentimen membantu kita memahami bagaimana masyarakat memandang 
        Karapan Sapi, memberikan wawasan berharga tentang praktik budaya dan dampaknya terhadap masyarakat.
""")
    

elif pages[selected_page] == 2:
    st.write("### Preprocessing")
    st.write("Here are the preprocessing results:")

    # Preprocessing functions
    def case_folding(text):
        """Mengubah teks menjadi huruf kecil."""
        if isinstance(text, str):
            return text.lower()
        return text

    def cleaning(text):
        """Menghapus karakter non-alfanumerik."""
        if isinstance(text, str):
            return re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text

    def tokenizing_spacy(text):
        """Tokenisasi menggunakan spaCy."""
        if isinstance(text, str):
            doc = nlp(text)
            return [token.text for token in doc]
        return text

    def remove_stopwords(tokens):
        """Menghapus stopwords menggunakan nltk."""
        if isinstance(tokens, list):
            stop_words = set(stopwords.words('english'))
            return [word for word in tokens if word.lower() not in stop_words]
        return tokens

    def lemmatization(tokens):
        """Lematisasi menggunakan nltk WordNetLemmatizer."""
        if isinstance(tokens, list):
            lemmatizer = WordNetLemmatizer()
            return [lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def load_normalization_dict(csv_file):
        """Memuat kamus normalisasi dari file CSV."""
        try:
            normalization_df = pd.read_csv(csv_file)
            if 'slang' in normalization_df.columns and 'formal' in normalization_df.columns:
                normalization_dict = normalization_df.set_index('slang')['formal'].to_dict()
                return normalization_dict
            else:
                raise ValueError("File kamus harus memiliki kolom 'slang' dan 'formal'.")
        except Exception as e:
            return None

    def normalization(text, normalization_dict):
        """Mengganti kata-kata dalam teks berdasarkan kamus normalisasi."""
        if isinstance(text, str):
            words = text.split()
            normalized_words = [normalization_dict[word] if word in normalization_dict else word for word in words]
            return ' '.join(normalized_words)
        return text

    # Main function to apply preprocessing
    def apply_preprocessing(df, selected_column, selected_step, normalization_dict=None):
    # Define preprocessing steps
        preprocessing_steps = {
            'Original': lambda x: x,  # Teks asli
            'Case Folding': case_folding,
            'Cleaning': cleaning,
            'Tokenizing': tokenizing_spacy,
            'Stopword Removal': remove_stopwords,
            'Lemmatization': lemmatization,
            'Normalization': lambda text: normalization(text, normalization_dict) if normalization_dict else text
        }

        # Apply preprocessing based on selected step
        if selected_step in ['Stopword Removal', 'Lemmatization', 'Normalization']:
            # If step requires tokenization, tokenize first
            df['Tokens'] = df[selected_column].apply(tokenizing_spacy)
            processed_column_name = f'{selected_step}'
            df[processed_column_name] = df['Tokens'].apply(preprocessing_steps[selected_step])
        else:
            # Apply the selected step directly
            processed_column_name = f'{selected_step}'
            df[processed_column_name] = df[selected_column].apply(preprocessing_steps[selected_step])

        return df, processed_column_name

    # File uploader
    uploaded_file = st.file_uploader("Pilih file CSV untuk Data", type="csv")
    normalization_file = st.file_uploader("Pilih file CSV untuk Kamus Normalisasi", type="csv")

    if uploaded_file is not None:
        try:
            # Read uploaded CSV file
            df = pd.read_csv(uploaded_file, encoding='utf-8', sep=';')

            # Show columns of the uploaded file
            st.write("### Kolom pada File yang Diunggah:")
            st.write(df.columns.tolist())

            # Dropdown to select the column for processing
            selected_column = st.selectbox("Pilih Kolom untuk Diproses", df.columns)
            st.write(f"Kolom yang dipilih: {selected_column}")
            # Convert selected column to string
            df[selected_column] = df[selected_column].astype(str)

            # If normalization file is uploaded, load normalization dictionary
            normalization_dict = None
            if normalization_file is not None:
                normalization_dict = load_normalization_dict(normalization_file)

            # Dropdown to select preprocessing step
            selected_step = st.selectbox(
                "Pilih Langkah Preprocessing",
                options=['Original', 'Case Folding', 'Cleaning', 'Tokenizing', 'Stopword Removal', 'Lemmatization', 'Normalization']
            )

            # Apply preprocessing based on selected step
            df, processed_column_name = apply_preprocessing(df, selected_column, selected_step, normalization_dict)

            # Show original and processed text
            st.write(f"### {selected_step} - Sebelum dan Sesudah")

            # Atur dataframe agar memenuhi lebar container dan menggunakan ukuran font yang lebih besar
            st.markdown("""
                <style>
                    .stDataFrame {
                        font-size: 16px !important;
                        overflow-x: auto;
                    }
                </style>
            """, unsafe_allow_html=True)

            st.dataframe(df[[selected_column, processed_column_name]].head(10))  # Pastikan menggunakan selected_column untuk tampilan yang benar

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file CSV: {e}")
    else:
        st.info("Silakan unggah file CSV untuk melanjutkan.")


    
elif pages[selected_page] == 3:
    st.write("### Word Cloud Page")
    
    image1 = Image.open("cluster0.png")
    image2 = Image.open("cluster1.png")

    image.show()
    # Membuat layout dua kolom
    col1, col2 = st.columns(2)

    with col1:
        st.image(image1, caption="Word Cloud 1", use_column_width=True)

    with col2:
        st.image(image2, caption="Word Cloud 2", use_column_width=True)

    # CSS tambahan untuk memastikan tidak ada garis bawah
    st.markdown("""
        <style>
        img {
            text-decoration: none !important;
            border: none !important;
        }
        </style>
        """, unsafe_allow_html=True)

    # Opsional: Penjelasan
    st.write("""
    Perbandingan antara dua word cloud:
    - **Word Cloud 1**: [Penjelasan tentang word cloud 1].
    - **Word Cloud 2**: [Penjelasan tentang word cloud 2].
    """)

elif pages[selected_page] == 4:
    st.write("Feature Extraction (TF-IDF)")
    st.write("TF-IDF Matrix:")
    st.dataframe(tfidf_df.head(10))

elif pages[selected_page] == 5:
    st.write("SMOTE Page")
    st.write("Resampled Training Features (X_train_resampled):")
    st.dataframe(X_train_resampled.head(10))
    st.write("Resampled Training Labels (y_train_resampled):")
    st.dataframe(y_train_resampled.head(10))

elif pages[selected_page] == 6:
    st.write("Model Performance Page")

    # Sample data for demonstration
    results = {
        'Model': ['K-means + SVM (5-Fold)', 'K-means + SMOTE + SVM (5-Fold)',
                  'K-means + SVM (10-Fold)', 'K-means + SMOTE + SVM (10-Fold)'],
        'Precision': [99.85, 99.70, 99.85, 99.85],
        'Recall': [99.84, 99.69, 99.84, 99.84],
        'F1-score': [99.84, 99.69, 99.84, 99.84],
        'Accuracy': [99.84, 99.69, 99.84, 99.84]
    }
    df_results = pd.DataFrame(results)
    st.write(df_results)

elif pages[selected_page] == 7:
    st.write("Test Page")
    user_input = st.text_area("Enter your text here")

    if st.button("Predict"):
        if user_input:  # Ensure input is not empty
            try:
                # Preprocess user input (if necessary)
                # For demonstration, only TF-IDF is applied here
                tfidf_vector = tfidf_vectorizer.transform([user_input])

                # Convert sparse matrix to dense array if necessary
                dense_tfidf_vector = tfidf_vector.toarray()

                # Perform prediction
                prediction = svm_classifier.predict(dense_tfidf_vector)
                st.write(f"Prediction: {prediction}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter some text to predict.")