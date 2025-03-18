from smolagents import agents, CodeAgent, LiteLLMModel, DuckDuckGoSearchTool
import os
import requests
import re
from markdownify import markdownify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool

# 初始化模型
model_id = "gemini/gemini-1.5-flash"
model = LiteLLMModel(model_id=model_id, token=os.getenv("GEMINI_API_KEY"))

# 讀取 PDF 檔案
file_path = "test.pdf"
try:
    pdf_reader = PdfReader(file_path)
    file_content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
except Exception as e:
    file_content = ""
    print(f"Error reading PDF: {e}")

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
        self.retriever = BM25Retriever.from_documents(docs, k=5)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Query must be a string."
        docs = self.retriever.invoke(query)
        return "\n".join([f"===== Document {i} =====\n{doc.page_content}" for i, doc in enumerate(docs)])

# 網頁訪問工具
class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = "Fetches webpage content as Markdown."
    inputs = {"url": {"type": "string", "description": "The URL to visit."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            markdown_content = markdownify(response.text).strip()
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
            return markdown_content[:10000]
        except requests.exceptions.Timeout:
            return "Request timed out. Try again later."
        except requests.exceptions.RequestException as e:
            return f"Error fetching the webpage: {str(e)}"

# 初始化工具
retriever_tool = RetrieverTool(docs_processed)
visit_webpage_tool = VisitWebpageTool()

# 建立 AI 代理
agent = CodeAgent(tools=[retriever_tool, visit_webpage_tool], model=model, add_base_tools=True)

# 執行查詢
query = "請以專業角度評論這份文件的結構、論證嚴謹性、完整性，並提供改進建議。"
retrieved_docs = retriever_tool.forward(query)

# 先讓 AI 總結文件，再進行分析
summary_query = "請總結以下文件的關鍵內容，保持簡潔清晰：\n" + retrieved_docs
summary = agent.run(summary_query)

final_query = summary + "\n\n" + query
response = agent.run(final_query)

print(response)