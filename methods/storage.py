from abc import ABC
from typing import Any, Union
from auth.config import GenAIConfig, RedisConfig
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings


default_redis: str = RedisConfig.REDIS_URL

default_vector: Any = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GenAIConfig.API_KEY
)


class VectorStorage(ABC):
    """
        Abstract base class for vector storage systems. Defines a method for storing text chunks along with their
        embeddings.
    """

    def create(self, text_chunks: list[str], **kwargs) -> Any:
        """
            Abstract method to be implemented by subclasses to store text chunks and their embeddings in a vector store.

            :param:
                - text_chunks: list[str
                - embeddings: str
                - kwargs:
        """
        ...

    def fetch(self, query: str) -> list[Any]:
        ...


class FAISS_Storage(VectorStorage, ABC):
    """
        Subclass of VectorStorage that implements vector storage using Redis. Handles the process of storing text
        chunks and their embeddings in a Redis vector store.
    """

    __slots__ = ("embeddings", "index_name")

    def __init__(self, **kwargs):
        self.index_name = kwargs.get("index_name", "pdf_chat")
        self.embeddings = kwargs.get("embeddings", default_vector)

    def create(self, text_chunks, **kwargs) -> VectorStoreRetriever:
        """
            Stores text chunks and their corresponding embeddings in Redis. Accepts an optional Redis URL via kwargs.
            Uses the default Redis URL if none is provided. Returns the vector data after storing it.

            :param:
                - text_chunks:
                - embeddings:
                - kwargs:

            :return:
                - RedisVectorStore
        """

        vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
        vector_store.save_local(
            folder_path="faiss",
            index_name=self.index_name
        )

        return vector_store.as_retriever()

    def fetch(self, query: str):
        new_db = FAISS.load_local(
            folder_path="faiss",
            index_name=self.index_name,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

        docs = new_db.similarity_search(query=query)

        return docs


allowed_storage_types = {
    "faiss": FAISS_Storage
}


def store_vector(text_chunks: list[str], embedding: Any, storage_type: str = "faiss", **kwargs) -> Any:
    """
        Function to store text chunks and embeddings in the specified storage type (e.g., Redis). Acts as a wrapper
        that dynamically selects the appropriate storage class based on the storage_type argument.

        :param
            - text_chunks:
            - embedding:
            - storage_type:
            - kwargs:

        :return: Any [VectorData]
    """
    if storage_type not in allowed_storage_types.keys():
        raise ValueError(f"Storage Type: {storage_type} not found")

    init_storage = allowed_storage_types.get(storage_type)(**kwargs)

    return init_storage.create(
        text_chunks=text_chunks,
        embeddings=embedding,
        **kwargs
    )


class RedisManager:
    connection = RedisConfig.connection()

    @classmethod
    def add(cls, key: str, data: str) -> bool:
        return cls.connection.set(name=key, value=data) if cls.connection else None

    @classmethod
    def fetch(cls, key: str) -> Union[str, None]:
        return cls.connection.get(name=key) if cls.connection else None
