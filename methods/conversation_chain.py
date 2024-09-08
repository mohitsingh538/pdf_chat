from abc import ABC, abstractmethod
from functools import lru_cache
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_redis import RedisVectorStore
from methods.storage import FAISS_Storage


class ConversationChain(ABC):

    @abstractmethod
    def get_conversation(self, question: str):
        ...


class HuggingFaceVectorChain(ConversationChain, ABC):

    def __init__(self, vector_data: RedisVectorStore, pdf_content: str):
        self.vector_data = vector_data
        self.pdf_content = pdf_content

    def get_conversation(self, question: str):
        from auth.config import HFConfig

        api_key = HFConfig.API_KEY
        if not api_key:
            ValueError("HUGGINGFACE API key is required. Set up in your .env file as HF_API_KEY=<YOUR_API_KEY")

        llm = HuggingFaceEndpoint(
            repo_id="hkunlp/instructor-xl",
            temperature=1,
            model_kwargs={"max_length": 512},
            huggingfacehub_api_token=api_key,
            timeout=600
        )

        # Initialize memory with PDF content as initial message
        memory = ConversationBufferMemory(memory_keys="pdf_chat_history", return_messages=True, output_key='answer')
        memory.save_context({"input": "PDF content"}, {"answer": self.pdf_content})

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_data.as_retriever(),
            memory=memory
        )

        # Get chat history
        chat_history = [
            {"role": "system", "content": "PDF content"},
            {"role": "system", "content": self.pdf_content}
        ]

        # Convert your chat history to a format expected by the model
        formatted_chat_history = {
            "past_user_inputs": [entry["content"] for entry in chat_history if entry["role"] == "human"],
            "generated_responses": [entry["content"] for entry in chat_history if entry["role"] == "system"]
        }

        # Generate response using the conversation chain
        response = conversation_chain({
            "question": question,
            "chat_history": formatted_chat_history
        })

        # Save the user's question and the model's response to memory
        memory.save_context({"input": question}, {"answer": response.get('result', 'No result found')})

        print("Response:", response.get('result', 'No result found'))

        return response.get('result', 'No result found')


class GeminiVectorChain(ConversationChain, ABC):

    def __init__(self, vector_data):
        self.vector_data = vector_data

    def get_conversation(self, question: str):
        from auth.config import GenAIConfig

        system_prompt = """
            Answer the question as detailed as possible from the provided context, make sure to provide all the 
            details, if the answer is not in provided context just say, "answer is not available in the context", don't 
            provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n
        """

        api_key = GenAIConfig.API_KEY
        if not api_key:
            raise ValueError("GEMINI API key is required. Set up in your .env file as GEMINI_API_KEY=<YOUR_API_KEY")

        model = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-pro",
            temperature=0.8,
            google_api_key=api_key
        )

        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["context", "question"]
        )

        chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
        docs = FAISS_Storage().fetch(question)

        response = chain.invoke(
            {"input_documents": docs, "question": question}, return_only_outputs=True
        )

        answer = response.get("output_text")

        print("response ==> ", answer)

        return answer
