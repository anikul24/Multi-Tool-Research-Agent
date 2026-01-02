from pinecone import Pinecone,ServerlessSpec
from semantic_router.encoders import OpenAIEncoder
from config import PINECONE_API_KEY, OPENAI_API_KEY, INDEX_NAME
import time

encoder = OpenAIEncoder(name='text-embedding-3-small', openai_api_key=OPENAI_API_KEY)


pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

pc = Pinecone(api_key=PINECONE_API_KEY)

##define serveless specification
spec = ServerlessSpec(
        cloud='aws',
        region='us-east-1'
)


def get_pinecone_index(index_name=INDEX_NAME):
    '''Get or create Pinecone index'''
    if index_name not in pc.list_indexes():
        print(f'Creating index: {index_name}')
        pc.create_index(
            name=index_name,
            dimension=1536, # Dimension for OpenAI text-embedding-3-small
            metric='cosine',
            serverless_spec=spec
        )
        # Wait for a few seconds to ensure the index is ready
        time.sleep(10)
    else:
        print(f'Index {index_name} already exists')
    
    index = pc.get_index(index_name)
    return index


def get_encoder():
    return encoder