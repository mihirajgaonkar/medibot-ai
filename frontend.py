#Step1 : Setup upload PDF functionality
import os
import streamlit as st 
from rag_pipeline import answer_query, retrieve_docs, llm_model

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# This stops Streamlit from watching all packages (optional)
os.environ["STREAMLIT_WATCH_DIRECTORIES"] = "false"

st.set_page_config(page_title="🧠 AI Medibot", layout="centered")

# === Sidebar ===
st.sidebar.title("📚 AI Medibot RAG System")
st.sidebar.write("Ask questions about uploaded documents using RAG and DeepSeek.")

# === Main Header ===
st.title("🩺 Medibot AI Chatbot")
st.markdown("Ask anything based on the uploaded PDF knowledge base!")

# === Chat Input ===
user_query = st.text_area("💬 Enter your question below:", height=150, placeholder="e.g. What is the investment philosophy of the author?")
ask_question = st.button("🧠 Ask Medibot")

# === Handle Query ===
if ask_question:
    if not user_query.strip():
        st.warning("⚠️ Please enter a question before submitting.")
    else:
        try:
            with st.spinner("Thinking..."):
                # Show user input
                st.chat_message("user").write(user_query)

                # Run RAG Pipeline
                think, answer = answer_query(query=user_query)

                # Show AI response (Answer)
                st.chat_message("assistant").write(answer)

                # Show <think> as expandable or light text
                if think:
                    with st.expander("🤖 Model reasoning", expanded=False):
                        st.markdown(f"<div style='color: gray; font-size: 0.9em'>{think}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")