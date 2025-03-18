import os
import requests
import re
import random
from markdownify import markdownify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from smolagents import agents, CodeAgent, LiteLLMModel, Tool

# 初始化模型
model_id = "gemini/gemini-1.5-flash"
model = LiteLLMModel(model_id=model_id, token=os.getenv("GEMINI_API_KEY"))

# 讀取 PDF 檔案
file_path = "test.pdf"
file_content = ""

try:
    pdf_reader = PdfReader(file_path)
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            file_content += f"\n\n===== Page {i + 1} =====\n{text}"
        else:
            print(f"⚠️ 頁面 {i + 1} 沒有文字，可能是圖片格式。")
except Exception as e:
    print(f"Error reading PDF: {e}")

# 確保有讀取到內容
if not file_content.strip():
    print("⚠️ PDF 沒有提取到任何文字，請檢查文件內容是否為可讀取的文字格式！")
    exit()

print("✅ PDF 內容成功讀取！前 1000 字預覽：")
print(file_content[:1000])

# 使用 TextSplitter 進行文本切割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_processed = [Document(page_content=text) for text in text_splitter.split_text(file_content)]

# 檢索工具類別
class RetrieverTool(Tool):
    name = "retriever"
    description = "Retrieves relevant document sections based on the query."
    inputs = {"query": {"type": "string", "description": "The query to perform."}}
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        k = min(len(docs), 5)
        self.retriever = BM25Retriever.from_documents(docs, k=k)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Query must be a string."
        docs = self.retriever.invoke(query)
        if not docs:
            return "❌ 找不到相關內容。"
        return "\n".join([f"===== 段落 {i+1} =====\n{doc.page_content}" for i, doc in enumerate(docs)])

# 初始化工具
retriever_tool = RetrieverTool(docs_processed)

# 建立 AI 代理（不同立場）
agent_a = CodeAgent(tools=[retriever_tool], model=model, add_base_tools=False)
agent_b = CodeAgent(tools=[retriever_tool], model=model, add_base_tools=False)

# 隨機分配立場
roles = ["支持", "反對"]
random.shuffle(roles)

def debate_round(agent_a, agent_b, topic, max_rounds=3):
    """
    讓兩個 Agent 進行辯論，並固定輪流發言
    """
    round_counter = 0
    statement = topic  # 初始問題

    while round_counter < max_rounds:
        print(f"🔵 Agent A（{roles[0]}方）: ")
        response_a = agent_a.run(f"{statement}\n請根據 PDF 提取相關內容進行論證。")
        print(response_a)

        print(f"🔴 Agent B（{roles[1]}方）: ")
        response_b = agent_b.run(f"{response_a}\n請反駁此觀點，並引用 PDF 內容作為證據。")
        print(response_b)

        statement = response_b  # 讓下一輪使用上一輪的觀點
        round_counter += 1

    print("💬 辯論結束！")

# 啟動辯論
debate_topic = "這份文件的論證結構是否嚴謹？"
debate_round(agent_a, agent_b, debate_topic)
