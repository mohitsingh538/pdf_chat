from typing import Any, Union
from abc import ABC, abstractmethod
from methods.storage import store_vector
from auth.config import GenAIConfig
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class VectorizeData(ABC):
    """
        An abstract base class defining the interface to vectorize data. It requires implementing classes to provide
        a method to vectorize a list of text chunks.
    """

    @abstractmethod
    def vectorize_data(self, text_chunks: list[str], **kwargs) -> Union[Any, None]:
        ...


class HFVectorize(VectorizeData, ABC):
    """
        A concrete implementation of VectorizeData using Hugging Face models for vectorization. This class provides
        methods to create and use Hugging Face embeddings for converting text chunks into vectors.
    """

    __DEFAULT_MODEL = "hkunlp/instructor-xl"

    @classmethod
    def __embedding_via_hugging_face(cls, **kwargs):
        """
            Creates and configures a Hugging Face embedding instance using the specified model and device. This method
            sets up the embedding configuration based on the availability of CUDA or CPU.
        """

        import torch
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings

        hf_config = {
            "model_name": kwargs.get("model_name", cls.__DEFAULT_MODEL),
            "model_kwargs": {'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            "encode_kwargs": {'normalize_embeddings': True}
        }

        return HuggingFaceEmbeddings(**hf_config)

    def vectorize_data(self, text_chunks, **kwargs):
        """
            Vectorize a list of text chunks using Hugging Face embeddings. It retrieves the embeddings, stores the
            vectorized data, and returns it. In case of an error, it prints an exception message and returns None.
        """

        try:
            embeddings = self.__embedding_via_hugging_face(**kwargs)
            if embeddings:
                vector_data = store_vector(text_chunks=text_chunks, embedding=embeddings)

                return vector_data

        except Exception as e:
            print("Exception occurred while vectorize", e)

        return None


class GeminiVectorizer(VectorizeData, ABC):
    """
        A concrete implementation of VectorizeData using Google's Gemini model for vectorization. This class provides
        methods to authenticate with Google Cloud and use the Gemini model to convert text chunks into vectors.
    """

    def vectorize_data(self, text_chunks, **kwargs):
        """
            Vectorize a list of text chunks using the Gemini model. It performs authentication, uses the model to
            generate embeddings, stores the vectorized data, and returns it. In case of an error, it prints an
            exception message and returns None
        """
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GenAIConfig.API_KEY
        )

        try:
            vector_data = store_vector(text_chunks=text_chunks, embedding=embeddings)

            return vector_data

        except Exception as e:
            print("Exception occurred while vectorize", e)

        return None


__allowed_vectorizers = {
    "huggingface": HFVectorize,
    "gemini": GeminiVectorizer
}


def get_vectorizer(text_chunks: list[str], vectorizer: str = "gemini", **kwargs):
    """
        Retrieves and uses a vectorizer based on the specified type. It validates the vectorizer type, initializes the
        corresponding vectorizer class, and calls its vectorize_data method to process the text chunks. Raises a
        ValueError if the vectorizer type is not allowed.
    """

    if vectorizer not in __allowed_vectorizers.keys():
        raise ValueError(f"Unable to use vectorizer: {vectorizer}. Allowed types: {list(__allowed_vectorizers.keys())}")

    return __allowed_vectorizers.get(vectorizer)().vectorize_data(text_chunks=text_chunks, **kwargs)
