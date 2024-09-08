import argparse
import warnings
from methods.pdf_reader import pdf_as_text
from methods.vectorize import get_vectorizer
from methods.split_text import get_text_chunks
from methods.conversation_chain import GeminiVectorChain


def pdf_to_vector(file_path: str):
    pdf_data = pdf_as_text(pdf_path=file_path)

    chunk_data = get_text_chunks(pdf_data)
    vector_data = get_vectorizer(text_chunks=chunk_data)

    return vector_data


def ask_questions(pdf_path):
    vector_data = pdf_to_vector(pdf_path)

    while True:
        query = input("Please enter your question: ")

        GeminiVectorChain(vector_data=vector_data).get_conversation(question=query)


if __name__ == "__main__":
    try:
        warnings.filterwarnings("ignore")

        parser = argparse.ArgumentParser(description="Process a PDF file.")
        parser.add_argument('--pdf_path', type=str, required=True, help="Path to the PDF file")

        args = parser.parse_args()
        pdf_file_path = args.pdf_path

        if not pdf_file_path:
            raise Exception("Please provide a valid PDF file path.")

        ask_questions(pdf_file_path)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting PDF Chat...")
        exit(0)
