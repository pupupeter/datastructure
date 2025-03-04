from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, HfApiModel
import os
import requests
import re
from markdownify import markdownify
from requests.exceptions import RequestException

# 📄 讀取 txt 檔案
file_path = "test.txt"
with open(file_path, "r", encoding="utf-8") as file:
    file_content = file.read()

# 📌 使用 TextSplitter 將長文本切分成多個小段落
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_processed = [Document(page_content=text) for text in text_splitter.split_text(file_content)]

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve relevant parts of the document."
    inputs = {"query": {"type": "string", "description": "The query to perform."}}
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=5)  # 取前5筆最相關的

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        docs = self.retriever.invoke(query)
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {i} =====\n{doc.page_content}" for i, doc in enumerate(docs)]
        )

class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = "Visits a webpage at the given URL and reads its content as a markdown string."
    inputs = {"url": {"type": "string", "description": "The URL of the webpage to visit."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        """Visits a webpage at the given URL and returns its content as a markdown string."""
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            markdown_content = markdownify(response.text).strip()
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
            return markdown_content[:10000]
        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"

# 🚀 初始化檢索工具
retriever_tool = RetrieverTool(docs_processed)
visit_webpage_tool = VisitWebpageTool()

# ✅ 設定 Hugging Face API Token
model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
model = HfApiModel(model_id=model_id, token=os.getenv("HF_API_TOKEN"))

# 🎯 讓 AI 透過 RAG 檢索 + 回答問題
agent = CodeAgent(tools=[retriever_tool, visit_webpage_tool], model=model, add_base_tools=True)

# 🔥 讓 AI 先檢索文件，然後回答問題
query = "請總結這份文件的重點，並提供最新的相關背景資訊"
retrieved_docs = retriever_tool.forward(query)
response = agent.run(retrieved_docs + "\n" + query)

print(response)
