from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
import pandas as pd
import torch
import gradio as gr

class LLAMA2:
    def __init__(self, csv_file, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.df = pd.read_csv(csv_file)
        self.df['combined_text'] = "Instruction: " + self.df["instruction"] + "\nResponse: " + self.df["response"]
        self.documents = self.df['combined_text'].tolist()

        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        # Initialize embedding model and vector store
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = FAISS.from_texts(self.documents, self.embedding_model)

        # Initialize Llama2 model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=self.quant_config).to("cuda")
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.95
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

        # Initialize message history
        self.history = ChatMessageHistory()
        self.history.add_ai_message('You are a helpful customer support agent. Use the context to provide clear, concise, and informative responses.')

    def get_context(self, query):
        retriever = self.db.as_retriever()
        retrieved_docs = retriever.get_relevant_documents(query, top=5)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        return context
    
    def get_response(self, query, history=None):
        context = self.get_context(query)
        self.history.add_user_message(query)
        
        prompt = f"""You are a helpful customer support agent. Use the following context to answer the user's query:
                        Context: {context}

                        User query: {query}

                        Answer the query based on the above context. If you cannot find relevant information, politely let the user know. Provide a clear and concise response:"""
       
        response = self.llm(prompt)    
        ai_response = response.strip()          
        self.history.add_ai_message(ai_response)

        return ai_response


if __name__ == "__main__":
    llm = LLAMA2('customer_support_dataset.csv')
    chatbot=gr.ChatInterface(
        fn=llm.get_response,
        type="messages",
    )
    chatbot.launch()

