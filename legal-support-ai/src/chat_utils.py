import os
import json
import uuid
import logging
from redis import Redis
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict
from pydantic import BaseModel
from typing import List,Optional
from dotenv import load_dotenv
import tempfile
from gtts import gTTS

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve variables
redis_host = os.getenv('REDIS_HOST')
redis_password = os.getenv('REDIS_PASSWORD')

# This is an alternative function to langchain redischatmessagehistory it bypasses the use of redis search
class WindowsRedisChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.redis =Redis(host=redis_host,port=6379,password=redis_password,ssl=True)
        self.session_id = session_id
        self.key = f"chat:{session_id}"

    def add_message(self, message: BaseMessage) -> None:
        self.redis.rpush(self.key, json.dumps(message_to_dict(message)))

    @property
    def messages(self) -> List[BaseMessage]:
        _items = self.redis.lrange(self.key, 0, -1)
        return messages_from_dict([json.loads(m.decode("utf-8")) for m in _items])

    def clear(self) -> None:
        self.redis.delete(self.key)




# Redis Session Manager
class RedisSessionManager:
    def __init__(self):
        self.redis = Redis(host=redis_host,port=6379,password=redis_password,ssl=True)
        self.session_ttl = 3600  # 24 hours
    def create_session(self, profile: dict) -> str:
        session_id = str(uuid.uuid4())
        self.redis.setex(
            f"session:{session_id}:profile",
            self.session_ttl,
            json.dumps(profile)
        )
        return session_id

    def get_profile(self, session_id: str) -> Optional[dict]:
        profile_data = self.redis.get(f"session:{session_id}:profile")
        return json.loads(profile_data) if profile_data else None

    def delete_session(self, session_id: str):
        keys = [f"session:{session_id}:profile", f"session:{session_id}:history"]
        self.redis.delete(*keys)

session_manager = RedisSessionManager()


# Initialize components
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

parser = StrOutputParser()

vector_store = PineconeVectorStore(
    index_name="legal-support-ai",
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

retriever = vector_store.as_retriever()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


def text_to_speech(text, lang='en', slow=False):
    """
    Convert text to speech and save as a temporary MP3 file.

    :param text: The text to convert to speech.
    :param lang: Language code (default is 'en' for English).
    :param slow: Boolean indicating whether the speech should be slow (default is False).
    :return: The path to the temporary MP3 file.
    """
    # Create a temporary file with a .mp3 suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        # Initialize gTTS with the provided text
        tts = gTTS(text=text, lang=lang, slow=slow)
        # Save the audio content to the temporary file
        tts.save(temp_file.name)
        # Return the path to the temporary file
        return temp_file.name


# System prompt remains largely the same but is part of ChatPromptTemplate
system_prompt_template = """You are a highly professional legal practitioner assistant named Amani.
Your primary role is to assist users in understanding legal concepts, interpreting case details, and navigating legal procedures with clarity, precision, and professionalism.
 You must always maintain a formal, respectful tone, and ensure legal information is accurate, clearly explained, and grounded in applicable frameworks or precedents.
Introduce yourself as Amani when first responding to a user. Always respond with authority, courtesy, and professionalism appropriate to a legal support specialist.
 Maintain conversation context based on the provided history.
 Ensure no to fabricate information; if unsure, advise the user to consult a qualified legal professional.
Retrieved relevant documents:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_template),
    MessagesPlaceholder(variable_name="history"), # Crucial for memory
    ("human", "{question}"),
])

# --- Create Runnable with Context and History ---
# This part combines retrieval with the chat model and history
runnable_with_context = RunnablePassthrough.assign(
    context=lambda x: retriever.invoke(x["question"]) # Retrieve context based on current question
) | prompt | llm | parser

# Function to get or create a WindowsRedisChatMessageHistory instance
def get_redis_history(session_id: str) -> BaseChatMessageHistory:
    return WindowsRedisChatMessageHistory(session_id)

# --- Create Conversational Chain with Memory ---
chain_with_history = RunnableWithMessageHistory(
    runnable_with_context,
    get_redis_history,
    input_messages_key="question", # This is the key for the user's input
    history_messages_key="history", # This is where Langchain will inject historical messages
)

# --- Updated Chat Interface ---
def chat_interface(question: str, session_id: str = "default_session"):
    """
    Invokes the conversational chain with the given question and session_id.
    """
    # The config dictionary is crucial for RunnableWithMessageHistory
    # It tells the chain which session_id to use for history.
    config = {"configurable": {"session_id": session_id}}
    result = chain_with_history.invoke({"question": question}, config=config)
    return result
