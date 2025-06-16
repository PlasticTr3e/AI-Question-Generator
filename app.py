import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import fitz
import tempfile
import os

# === Konfigurasi Halaman ===
st.set_page_config(page_title="QuizBot", page_icon="🧠", layout="wide")

# === Custom CSS ===
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(to right, #fceabb, #f8b500);
            color: #2c3e50;
        }

        .title {
            text-align: center;
            font-size: 48px;
            color: #0d47a1;
            font-weight: 700;
            margin-top: 30px;
        }

        .description {
            font-size: 20px;
            text-align: center;
            color: #37474f;
            margin-bottom: 40px;
        }

        .stContainer {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 25px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            margin-bottom: 30px;
        }

        .stButton button {
            background-color: #ec407a;
            color: white;
            padding: 12px 28px;
            font-size: 18px;
            border-radius: 12px;
            font-weight: 600;
            border: none;
        }

        .stButton button:hover {
            background-color: #c2185b;
        }

        .footer-text {
            font-size: 16px;
            text-align: center;
            color: #546e7a;
            margin-top: 40px;
        }

        ul.question-list {
            padding-left: 20px;
            margin-top: 15px;
        }

        ul.question-list li {
            margin-bottom: 10px;
            font-size: 17px;
            color: #1a237e;
        }
    </style>
""", unsafe_allow_html=True)

# === Load Model ===
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("t5-base-multi-qg-squadv2")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base-multi-qg-squadv2")
    return tokenizer, model

# === PDF Extractor ===
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.getbuffer())
        temp_pdf_path = temp_file.name

    doc = None
    try:
        doc = fitz.open(temp_pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text("text")
        return text
    finally:
        if doc:
            doc.close()
        os.remove(temp_pdf_path)

# === Question Generator ===
def generate_questions(text):
    tokenizer, model = load_model()
    prefix = "generate questions: "
    inputs = tokenizer(prefix + text, return_tensors="pt", max_length=2048, truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.80,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    questions_raw = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return [q.strip() for q in questions_raw.replace("sep>", "\n").split("\n") if q.strip()]

# === Main App ===
def main():
    if 'started' not in st.session_state:
        st.session_state.started = False

    st.markdown('<div class="title">QuizBot: AI Question Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Buat pertanyaan otomatis dari teks bahasa Inggris 🎓</div>', unsafe_allow_html=True)

    if not st.session_state.started:
        if st.button("Mulai Sekarang", use_container_width=True):
            st.session_state.started = True

    if st.session_state.started:
        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.subheader("Input Teks")
                input_method = st.selectbox("Metode input:", ["Upload File", "Manual Input"])
                text_content = ""

                if input_method == "Upload File":
                    uploaded_file = st.file_uploader("Unggah file .txt atau .pdf", type=["txt", "pdf"])
                    if uploaded_file:
                        ext = uploaded_file.name.split('.')[-1].lower()
                        if ext == "txt":
                            text_content = uploaded_file.read().decode("utf-8")
                            st.text_area("Isi file:", text_content, height=200, disabled=True)
                        elif ext == "pdf":
                            text_content = extract_text_from_pdf(uploaded_file)
                            st.text_area("Isi PDF:", text_content, height=200, disabled=True)
                        else:
                            st.error("Hanya file .txt dan .pdf yang didukung.")
                else:
                    text_content = st.text_area("Masukkan teks di sini:", height=200)

                generate = st.button("✨ Buat Pertanyaan", use_container_width=True)

        with col2:
            with st.container(border=True):
                st.subheader("Hasil Pertanyaan")
                if generate and text_content:
                    with st.spinner("🔄 Sedang memproses..."):
                        try:
                            questions = generate_questions(text_content)
                            st.success("Pertanyaan berhasil dibuat!")
                            st.markdown("<ul class='question-list'>", unsafe_allow_html=True)
                            for q in questions:
                                st.markdown(f"<li>{q}</li>", unsafe_allow_html=True)
                            st.markdown("</ul>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Terjadi kesalahan: {e}")
                elif generate and not text_content:
                    st.warning("Masukkan teks terlebih dahulu!")

    else:
        st.markdown('<div class="footer-text">Klik tombol "Mulai Sekarang" untuk menggunakan QuizBot ✨</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
