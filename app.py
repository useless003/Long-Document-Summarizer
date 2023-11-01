import streamlit as st
from PyPDF2 import PdfReader
import nltk

nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

nltk.download("punkt")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text = text+page.extract_text()
    return text

def text_summarizer(text, num_sentences):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Calculate word frequency
    word_frequency = FreqDist(filtered_words)

    # Assign scores to each sentence based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequency:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_frequency[word]
                else:
                    sentence_scores[sentence] += word_frequency[word]

    # Sort sentences based on their scores
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

    # Select the top 'num_sentences' sentences to form the summary
    summary_sentences = [sentence for sentence, _ in sorted_sentences[:num_sentences]]

    return " ".join(summary_sentences)

if __name__ == "__main__":
    st.title("Large Document Summarizer📑")

    st.markdown("")
    st.markdown("")

    file = st.file_uploader("Upload your document here(:red[! PDF only])", accept_multiple_files=True)

    st.markdown("")
    st.markdown("")

    range = st.slider("Choose the range of the summarization", value=50, min_value=10, max_value=1000, step=10)
    # Get user input for the text
    text = get_pdf_text(file)
    st.markdown("")
    st.markdown("")
    summarize = st.button("Summarize")
    if summarize and file and range:
        summary = text_summarizer(text, range)
        st.subheader("Your Summarized Text")
        st.write(summary)

    else:
        st.warning(":red[Please provide all the necessary inputs!]")


