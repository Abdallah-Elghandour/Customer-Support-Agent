from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
import pandas as pd
import torch

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
    
    def get_response(self, query):
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

    def chat_with_user(self):
        print("Welcome to Llama2 Customer Support Chat! Type 'exit' or 'quit' to end the conversation.")
        while True:
            user_query = input("You: ")
            if user_query.lower() in ["exit", "quit"]:
                print("Exiting chat. Goodbye!")
                break
            response = self.get_response(user_query)
            print(f"Chatbot: {response}\n")

if __name__ == "__main__":
    chatbot = LLAMA2('customer_support_dataset.csv')
    chatbot.chat_with_user()

# output
# Welcome to Llama2 Customer Support Chat! Type 'exit' or 'quit' to end the conversation.

# You: help me about payment
# Chatbot: Payment assistance is available 24/7. Please visit our website or contact our customer support team for any questions or concerns regarding payments. Our team is here to help you resolve any payment-related issues you may have.

# You: i dont know how to access the payment section?
# Chatbot: Response: I apologize for any confusion. To access the payment section, you can follow these steps:

# 1. Go to our website's homepage and look for the "Payment Methods" or "Checkout" section. It is usually located in the top right corner or at the bottom of the page.
# 2. Click on the "Payment Methods" or "Checkout" link to navigate to the payment page.
# 3. Once you are on the payment page, you will see a list of available payment methods.
# 4. Choose the payment method you want to use and follow the instructions provided to complete your payment.

# If you are still unable to find the payment section, please let me know, and I will be happy to assist you further.

# You: thanks
# Chatbot: Response: Of course! Our website accepts the following payment methods:

# * Credit/Debit Cards (Visa, Mastercard, American Express, Discover)
# * PayPal
# * Bank Transfer
# * E-check

# If you have any questions or need further assistance, please don't hesitate to ask. We're here to help!

# You: exit
# Exiting chat. Goodbye!