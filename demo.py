import streamlit as st
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


hf = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 50},
)
from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: """
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

def mistral(question):
    return chain.invoke({"question": question})


def main():
    st.title("Greeting Application")
    question = st.text_input("Enter your name", "")
    result=mistral(question)
    if question:
        st.write(result)

if __name__ == "__main__":
    main()
