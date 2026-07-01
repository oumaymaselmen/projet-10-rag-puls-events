import langchain
import faiss
import mistralai
import pandas
import tiktoken
from dotenv import load_dotenv

print(" langchain :", langchain.__version__)
print(" faiss : OK")
print(" mistralai :", mistralai.__version__)
print(" pandas :", pandas.__version__)
print(" tiktoken :", tiktoken.__version__)
print(" python-dotenv : OK")
print("\n Environnement prêt !")