from openai import OpenAI
import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import datetime
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import re


# openai API키 입력
load_dotenv()

client = OpenAI(
    api_key = os.getenv("OPENAI_API_KEY")
)

model = ChatOpenAI(model="gpt-4o-mini")

# csv 파일 로드.
loader = CSVLoader('Merged_details.csv',encoding='UTF8')

videos = loader.load()


recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=5,
    length_function=len,
    is_separator_regex=False,
)

splits = recursive_text_splitter.split_documents(videos)


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# if os.path.exists('./db/faiss'):
#     vectorstore = FAISS.load_local('./db/faiss', embeddings, allow_dangerous_deserialization=True)
# else:
#     vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
#     vectorstore.save_local('./db/faiss')
    
vectorstore = FAISS.from_documents(documents=videos, embedding=embeddings)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})


# 프롬프트 템플릿 정의
contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", "Please respond only in Korean. Recommend 5 videos that the user would like to see recommended among dramas, movies, and animations. Search and recommend keywords in the Genre list in Merged_details.csv in priority order."),
    ("user", "Context: {context}\\n\\nQuestion: {question}. Provide clear and specific recommendations based on the data.")

])




# 디버깅을 위해 만든 클래스
class SimplePassThrough:
    def invoke(self, inputs, **kwargs):
        return inputs

# 프롬프트 클래스
class ContextToPrompt:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    def invoke(self, inputs):
        # response_docs 내용을 trim (가독성을 높여줌)
        if isinstance(inputs, list): # inputs가 list인 경우. 즉 여러개의 문서들이 검색되어 리스트로 전달된 경우
            context_text = "\n".join([doc.page_content for doc in inputs]) # \n을 구분자로 넣어서 한 문자열로 합치기
        else:
            context_text = inputs # 리스트가 아닌경우는 그냥 리턴

        # 프롬프트
        formatted_prompt = self.prompt_template.format_messages( # 템플릿의 변수에 삽입
            context=context_text, # {context} 변수에 context_text, 즉 검색된 문서 내용을 삽입
            question=inputs.get("question", "")
        )
        return formatted_prompt

# Retriever 클래스
class RetrieverWrapper:
    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(self, inputs):
        # 0단계 : query의 타입에 따른 전처리
        if isinstance(inputs, dict): # inputs가 딕셔너리 타입일경우, question 키의 값을 검색 쿼리로 사용
            query = inputs.get("question", "")
        else: # 질문이 문자열로 주어지면, 그대로 검색 쿼리로 사용
            query = inputs
        # 1단계 : query를 리트리버에 넣어주고, response_docs를 얻기
        response_docs = self.retriever.get_relevant_documents(query) # 검색을 수행하고 검색 결과를 response_docs에 저장
        return response_docs

# RAG 체인 설정
rag_chain_debug = {
    "context": RetrieverWrapper(retriever), # 클래스 객체를 생성해서 value로 넣어줌
    "prompt": ContextToPrompt(contextual_prompt),
    "llm": model
}

# 챗봇 구동
while True:
    print("========================")
    query = input("질문을 입력하세요 : ")
    
    # 1. Retriever로 관련 문서 검색
    response_docs = rag_chain_debug["context"].invoke({"question": query})
    
    # 2. 문서를 프롬프트로 변환
    prompt_messages = rag_chain_debug["prompt"].invoke({
        "context": response_docs,
        "question": query
    })
    
    # 3. LLM으로 응답 생성
    response = rag_chain_debug["llm"].invoke(prompt_messages)
    
    print("\n답변:")
    print(response.content)
