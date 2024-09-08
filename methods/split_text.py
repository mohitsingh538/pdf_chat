import json
from typing import Any, Union
from methods.pdf_reader import hash_pdf_data
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from methods.storage import RedisManager


def get_text_chunks(pdf_content: str, splitter_type: Any = CharacterTextSplitter) -> Union[list[str], str, None]:
    cache_key = hash_pdf_data(pdf_content, splitter_type)

    cached_data = RedisManager.fetch(cache_key)
    if cached_data is not None:
        print("Fetched pdf_hash from redis")
        return json.loads(cached_data)

    try:
        content_split = splitter_type(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = content_split.split_text(pdf_content)

        try:
            RedisManager.add(cache_key, json.dumps(chunks))

        except Exception as e:
            print(f"Exception in adding chunks to redis ==> {e}")

        return chunks

    except Exception as e:
        print(f"Unable to split text using class {splitter_type}.\nException ==> {e}\nTry again with another class.")

        return None
