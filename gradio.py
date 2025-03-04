import gradio as gr
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
from PyPDF2 import PdfReader

def process_pdf(file):
    pdf_reader = PdfReader(file.name)
    file_content = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_processed = [Document(page_content=text) for text in text_splitter.split_text(file_content)]
    return docs_processed

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve relevant parts of the document."
    inputs = {"query": {"type": "string", "description": "The query to perform."}}
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=5)

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

def query_pdf(pdf_file, query):
    docs_processed = process_pdf(pdf_file)
    retriever_tool = RetrieverTool(docs_processed)
    visit_webpage_tool = VisitWebpageTool()
    model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
    model = HfApiModel(model_id=model_id, token=os.getenv("HF_API_TOKEN"))
    agent = CodeAgent(tools=[retriever_tool, visit_webpage_tool], model=model, add_base_tools=True)
    retrieved_docs = retriever_tool.forward(query)
    response = agent.run(retrieved_docs + "\n" + query)
    return response

iface = gr.Interface(
    fn=query_pdf,
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Enter your query")],
    outputs="text",
    title="PDF Query with RAG",
    description="Upload a PDF and enter a query to retrieve relevant content."
)

iface.launch()