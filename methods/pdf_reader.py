import hashlib
import pickle
from functools import lru_cache
from typing import Any
from PyPDF2 import PdfReader


@lru_cache(maxsize=512)
def pdf_as_text(pdf_path: str):
    content = ""

    reader = PdfReader(pdf_path).pages
    for page in reader:
        page_data = page.extract_text()
        content += page_data

    return content


def hash_pdf_data(pdf_content: str, splitter_type: Any) -> str:
    hasher = hashlib.md5()
    hasher.update(pdf_content.encode('utf-8'))
    hasher.update(pickle.dumps(splitter_type))

    return hasher.hexdigest()

