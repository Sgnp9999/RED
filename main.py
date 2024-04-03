from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


hf = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
)
from langchain_core.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

def main():
    while True:
        question = input("Ask Q: ")
        print(chain.invoke({"question": question}))

if __name__ =="__main__":
    print("calling")
    main()