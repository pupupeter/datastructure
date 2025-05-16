import os
import random
import requests
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, LiteLLMModel, Tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from duckduckgo_search import DDGS
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
class DuckDuckGoSearchTool(Tool):
    name = "ddg_search"
    description = "使用 DuckDuckGo 進行網路搜尋"
    inputs = {"query": {"type": "string", "description": "要搜尋的關鍵字"}}
    output_type = "string"
    def forward(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=3)]
            return "\n".join([f"標題：{r['title']}\n內容：{r['body']}" for r in results])
        except Exception as e:
            return f"❌ 搜尋失敗：{str(e)}"
model_id = "gemini/gemini-2.0-flash"
model = LiteLLMModel(model_id=model_id, token=os.getenv("GEMINI_API_KEY"))
def load_pdf(file_path):
    content = ""
    try:
        pdf_reader = PdfReader(file_path)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                content += f"\n\n===== Page {i + 1} =====\n{text}"
    except Exception as e:
        return f"❌ PDF 讀取失敗：{e}", None
    if not content.strip():
        return "❌ PDF 無法提取任何文字內容。", None
    return "✅ PDF 讀取成功！", content
def fetch_wikipedia_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return "\n".join([p.text.strip() for p in paragraphs if p.text.strip()])
    except Exception as e:
        return f"❌ 無法載入 Wikipedia 內容：{e}"
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
        docs = self.retriever.invoke(query)
        if not docs:
            return "❌ 找不到相關內容。"
        return "\n".join([f"===== 段落 {i+1} =====\n{doc.page_content}" for i, doc in enumerate(docs)])
def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)
def start_debate(file_path, topic, rounds, use_theory):
    status, content = load_pdf(file_path)
    if not content:
        return status
    theory_snippet = ""
    if use_theory:
        theory_text = fetch_wikipedia_text("https://zh.wikipedia.org/zh-tw/%E6%96%B0%E5%BC%8F%E5%A5%A7%E5%8B%92%E5%B2%A1%E5%88%B6")
        if theory_text.startswith("❌"):
            return theory_text
        theory_snippet = f" 理論參考（選用）：\n{theory_text[:1000]}...\n\n"
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=text) for text in splitter.split_text(content)]
    retriever_tool = RetrieverTool(docs)
    ddg_tool = DuckDuckGoSearchTool()
    agent_a = CodeAgent(tools=[retriever_tool, ddg_tool], model=model, add_base_tools=False)
    agent_b = CodeAgent(tools=[retriever_tool, ddg_tool], model=model, add_base_tools=False)
    roles = ["支持", "反對"]
    random.shuffle(roles)
    debate_log = f" 主題：{topic}\n Agent A 是 {roles[0]} 方，Agent B 是 {roles[1]} 方\n\n"
    statement = topic
    for i in range(rounds):
        related_text_a = retriever_tool.forward(query=statement)
        prompt_a = (
            f"你是辯論的{roles[0]}方。\n"
            f"請根據下方 PDF 提取的相關段落，以及必要時可以使用 ddg_search 工具查網路資料，來支持此主張：{statement}\n\n"
            f" PDF 相關段落：\n{related_text_a}\n\n"
            f"{theory_snippet}"
        )
        response_a = agent_a.run(prompt_a)
        sentiment_a = analyze_sentiment_vader(response_a)
        debate_log += f"\n Round {i+1} - Agent A（{roles[0]}）:\n{response_a}\n情感分析:{sentiment_a}\n"
        if sentiment_a['neg'] == 1.0 or sentiment_a['pos'] == 1.0:
            debate_log += f"\n 裁判：Agent A（{roles[0]}）因情感過於極端而判輸！\n"
            return debate_log
        related_text_b = retriever_tool.forward(query=response_a)
        prompt_b = (
            f"你是辯論的{roles[1]}方。\n"
            f"請根據 PDF 中的內容反駁以下觀點，必要時可以使用 ddg_search 工具查網路資料：\n{response_a}\n\n"
            f" PDF 相關段落：\n{related_text_b}\n\n"
            f"{theory_snippet}"
        )
        response_b = agent_b.run(prompt_b)
        sentiment_b = analyze_sentiment_vader(response_b)
        debate_log += f"\n Round {i+1} - Agent B（{roles[1]}）:\n{response_b}\n情感分析:{sentiment_b}\n"
        if sentiment_b['neg'] == 1.0 or sentiment_b['pos'] == 1.0:
            debate_log += f"\n 裁判：Agent B（{roles[1]}）因情感過於極端而判輸！\n"
            return debate_log
        statement = response_b
    debate_log += "\n 辯論結束！"
    return debate_log
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        topic = request.form["topic"]
        rounds = int(request.form["rounds"])
        use_theory = "use_theory" in request.form
        file = request.files["pdf_file"]
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        debate_result = start_debate(file_path, topic, rounds, use_theory)
        return render_template("result.html", result=debate_result)
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)
