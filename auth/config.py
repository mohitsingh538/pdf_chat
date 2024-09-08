import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


class GenAIConfig:

    API_KEY = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=API_KEY)


class HFConfig:

    API_KEY = os.getenv('HF_API_KEY')


class RedisConfig:

    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = os.getenv("REDIS_PORT", 9560)
    db: int = os.getenv("REDIS_DB", 0)
    user: str = os.getenv("REDIS_USER")
    password: str = os.getenv("REDIS_PASSWORD")

    REDIS_URL: str = "redis://"

    if user and password:
        REDIS_URL += f"{user}:{password}@"

    REDIS_URL += f"{host}:{port}/{db}"

    @classmethod
    def connection(cls):
        import redis

        try:
            redis_conn: redis.Redis = redis.from_url(cls.REDIS_URL)

            return redis_conn

        except (redis.exceptions.ConnectionError, redis.exceptions.AuthenticationError, Exception) as e:
            raise e
