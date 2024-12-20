import os
import streamlit as st
import openai
import random
import pandas as pd
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, PromptHelper, ServiceContext
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.settings import Settings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough
)
from langchain_community.chat_message_histories import ChatMessageHistory

from typing import Optional, Type
from typing import List, OrderedDict
import requests
import json
import time

import os
import streamlit as st
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.schema import TextNode
from llama_index.core import PromptHelper
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.settings import Settings

from langsmith import Client
from langsmith import traceable