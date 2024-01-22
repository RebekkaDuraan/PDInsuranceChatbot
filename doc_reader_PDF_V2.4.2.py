
# SET ENVIRONMENT VARIABLE
import getpass
import os
# my_var = os.getenv('OPENAI_API_KEY')
# print(my_var)  # This will print 'my_value'
# os.environ["OPENAI_API_KEY"] = getpass.getpass()
from dotenv import load_dotenv
load_dotenv(r'doc_reader_PDF_Environment_Var.env')  # This will load the environment variables from the .env file

#create a simple indexing pipeline and RAG chain
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
import glob
import PyPDF2
import streamlit as st


class Document:
    def __init__(self, page_content, metadata={}):
        self.page_content = page_content
        self.metadata = metadata


class PDFLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    @st.cache_resource 
    def load(_self):
        docs = []
        pdf_files = glob.glob(os.path.join(_self.folder_path, '*.pdf'))

        for path in pdf_files:
            with open(path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    # Create a Document object
                    doc = Document(page_content=text, metadata={})
                    docs.append(doc)
        return docs


folder_path = "" #Relative Path
loader = PDFLoader(folder_path=folder_path)
docs = loader.load()


@st.cache_resource 
def initialize_model_and_vectorstore(_docs):
    # Split the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(_docs)

    # Check if splits are empty
    if not splits:
        raise ValueError("No document splits were generated. Check your document format and splitter settings.")

    # Generate dummy IDs for each split if they are not provided
    ids = [f"doc_{i}" for i in range(len(splits))]

    # Initialize vector store and model
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), ids=ids)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    return vectorstore, llm

vectorstore, llm = initialize_model_and_vectorstore(docs)
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()


prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Class to maintain conversation history
class Conversation:
    def __init__(self):
        self.history = []

    def add_exchange(self, question, answer):
        self.history.append((question, answer))

    def get_formatted_history(self):
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in self.history)
    
    def truncate_history(self, max_length):
        if len(self.history) > max_length:
            self.history = self.history[-max_length:]

# Instance of the Conversation class
conversation = Conversation()

# Modified custom prompt function to include conversation history
def custom_prompt(question):
    history = conversation.get_formatted_history()
    instructions = (
        """
        You are a friendly and helpful assistant, specialized in PD Insurance for cats and dogs. 
        Answer questions based on the content of the provided PDF. If the information is not in the PDF, 
        use the default answer. Do not mention the absence of information in the PDF. 
        For non-related queries, remind the user that the focus is on pet insurance. 
        In case of ambiguity, ask clarifying questions.
        Do not answer any questions about costs or prices, use the default answer.
        
        Default answer: "I'm sorry, I cannot help you with that. Please contact PD Insurance at 0800 738 467 or email contactus@pd.co.nz for more information or ask another question."
        """
    )
    # Ensure the history is correctly appended
    full_prompt = f"{instructions}\n\n{history}\n\nQuestion: {question}"
    # print("Current Prompt:", full_prompt)  # Debugging print
    return full_prompt

# Modify the rag_chain to include the custom prompt
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough() | custom_prompt}
    | prompt
    | llm
    | StrOutputParser()
)

# Adjust the ask_question function to include only recent history if needed
def ask_question(question):
    # Check for non-specific questions and handle them
    if question.strip().lower() in ["hi", "hello", "hey"]:
        return "Hello! How can I assist you with your pet insurance queries today?"

    # Truncate history if needed
    conversation.truncate_history(max_length=10)  # Keep the last 10 exchanges
    answer = rag_chain.invoke(question)

    # Check if the answer is satisfactory, else provide a default response
    if "unable to fully assist you" in answer:
        return "I'm here to help with specific questions about pet insurance. Could you please provide more details or ask a different question?"
    
    # Validate and modify response if necessary
    if "not provided in the given context" in answer:
        answer = "It seems I'm unable to fully assist you with this query. Would you like to speak with a human representative for more help?"

    conversation.add_exchange(question, answer)
    return answer

def ask_multiple_questions(questions):
    responses = []
    for question in questions:
        question = question.strip()  # Strip whitespace from each question
        if not question:
            continue  # Skip empty questions
        # Check for non-specific questions and handle them
        if question.lower() in ["hi", "hello", "hey"]:
            response = "Hello! How can I assist you with your pet insurance queries today?"
        else:
            # Truncate history if needed
            conversation.truncate_history(max_length=10)  # Keep the last 10 exchanges
            response = rag_chain.invoke(question)

            # Check if the answer is satisfactory, else provide a default response
            if "unable to fully assist you" in response:
                response = "I'm here to help with specific questions about pet insurance. Could you please provide more details or ask a different question?"
            
            # Validate and modify response if necessary
            if "not provided in the given context" in response:
                response = "It seems I'm unable to fully assist you with this query. Would you like to speak with a human representative for more help?"

        conversation.add_exchange(question, response)
        responses.append(response)
    return responses



# def main():
#     print("Welcome to the Pet Insurance Chatbot!")
#     print("Type 'quit' to exit the chat.")

#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == 'quit':
#             # print(f"You: {user_input}")  # Print the user's question
#             print("Chatbot: Thank you for using the Pet Insurance Chatbot. Have a great day!")
#             break

#         response = ask_question(user_input)
#         # print(f"You: {user_input}")  # Print the user's question
#         print(f"Chatbot: {response}")  # Print the chatbot's response

# if __name__ == "__main__":
#     main()

# Streamlit interface for the chatbot
# Title of the app
st.title('PD Insurance Chatbot')

# Initialize session state for conversation history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Display the conversation history
for index, (role, line) in enumerate(st.session_state['history']):
    # Use the index to create a unique key for each text area
    st.text_area(role, value=line, height=75, disabled=True, key=f"{role}_{index}")

# Function for Handling User Input
def handle_user_input():
    user_input = st.session_state.user_input
    if user_input.lower() == 'quit':
        st.session_state['history'] = []
        st.write("Chatbot: Thank you for using the Pet Insurance Chatbot. Have a great day!")
    else:
        # Get the response from the chatbot and update conversation history
        response = ask_question(user_input)  # Assuming 'ask_question' is a defined function
        st.session_state['history'].append(('You', user_input))
        st.session_state['history'].append(('Chatbot', response))
    
    # Clear the input field after processing the question
    st.session_state.user_input = ""

# Text input for user query at the bottom
user_input = st.text_input("Type your question here:", key="user_input", on_change=handle_user_input)

# Check if there is input and handle it (Optional: This block can be removed if the input is only handled in on_change)
if user_input:
    handle_user_input()