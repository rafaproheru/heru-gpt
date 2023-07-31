import os
from dotenv import load_dotenv, find_dotenv
import pinecone
import random
load_dotenv(find_dotenv(), override=True)

pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV')) 



nombre_index = 'heru-gpt'
index = pinecone.Index(nombre_index)

index.delete(delete_all=True)
print(index.describe_index_stats())