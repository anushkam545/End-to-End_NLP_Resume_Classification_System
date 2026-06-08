# --------------------------------------------
# Resume Job Category Classifier - Streamlit App
# --------------------------------------------

import streamlit as st
from prediction import predict_job_category

# Page Configuration

st.set_page_config(
    page_title="Resume Job Category Classifier",
    page_icon="📄",
    layout="centered"
)
 
# Custom CSS

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1E88E5;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6E6F71;
    margin-bottom: 20px;
}

.result-box {
    background-color: #F5F7FA;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
    border: 1px solid #E0E0E0;
    color: #000000;
}

.result-box h2 {
    color: #1E88E5;
    font-weight: bold;
}

.result-box h3 {
    color: #333333;
}

.stButton > button {
    width: 100%;
    height: 3rem;
    font-size: 18px;
    font-weight: 600;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# Sidebar

with st.sidebar:

    st.header("📌 About")

    st.write("""
    This application predicts the most suitable
    job category from a resume using a fine-tuned
    DistilBERT model.
    """)

    st.subheader("Tech Stack")

    st.write("• Python")
    st.write("• PyTorch")
    st.write("• Hugging Face Transformers")
    st.write("• DistilBERT")
    st.write("• Streamlit")

# Header

st.markdown(
    '<div class="title">📄 Resume Job Category Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict the job category using DistilBERT</div>',
    unsafe_allow_html=True
)

st.divider()
 
# File Upload

uploaded_file = st.file_uploader(
    "Upload Resume (.txt)",
    type=["txt"]
)

resume_text = ""

if uploaded_file is not None:
    resume_text = uploaded_file.read().decode("utf-8")

# Text Input

resume_text = st.text_area(
    "Paste Resume Text",
    value=resume_text,
    height=250,
    placeholder="Paste resume content here..."
)

# Prediction

if st.button("🔍 Predict Category"):

    if not resume_text.strip():

        st.warning("Please upload or paste a resume.")

    else:

        with st.spinner("Analyzing Resume..."):

            category, confidence = predict_job_category(
                resume_text
            )

        st.success("Prediction Completed")

        st.markdown(
            f"""
            <div class="result-box">
                <h3>Predicted Category</h3>
                <h2>{category}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "Confidence Score",
            f"{confidence:.2%}"
        )

