import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Page Title
st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="wide")

# SIDEBAR 
with st.sidebar:
    st.title("📊 Model Info")
    st.divider()

    st.subheader("Model Selected")
    st.success("Support Vector Machine (SVM)")

    st.divider()

    st.subheader("📈 Model Performance")
    st.metric(label="🎯 Accuracy", value="97.75%")
    st.divider()

    st.subheader("📂 Dataset Info")
    st.write("-- Total emails : 5572")
    st.write("-- Ham emails   : 4825")
    st.write("-- Spam emails  : 747")
    st.write("-- Ham %        : 86.6%")
    st.write("-- Spam %       : 13.4%")

    st.divider()

    st.subheader("⚙️ How It Works")
    st.write("1. Email text is cleaned")
    st.write("2. Converted to numbers using TF-IDF")
    st.write("3. SVM model predicts spam or ham")

# MAIN PAGE 
st.title("📧 Email Spam Detector")
st.write("Enter an email message below and check whether it is Spam or Not Spam.")

st.divider()

# USER INPUT 
email_text = st.text_area(
    "Enter Email Content",
    height=200,
    placeholder="Type or paste an email here..."
)

if st.button("Detect"):

    if email_text.strip() == "":
        st.warning("Please enter an email message.")
    else:
        # Transform input
        cleaned_text = clean(email_text)
        transformed_text = vectorizer.transform([cleaned_text])
        
        # Predict
        detection = model.predict(transformed_text)

        if detection[0] == "spam":
            st.error("🚫 Spam Email")
        else:
            st.success("✅ Not Spam Email")

# KEY INSIGHTS 
st.subheader("💡 Key Insights")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("""
    **📊 Dataset Imbalance**

    86.6% of emails are Ham and
    only 13.4% are Spam. This
    reflects real world email
    distribution.
    """)

with col2:
    st.info("""
    **🏆 Why SVM?**

    SVM caught 129 out of 150
    spam emails vs Naive Bayes
    which caught only 120.
    SVM missed 9 fewer spams.
    """)

st.divider()
