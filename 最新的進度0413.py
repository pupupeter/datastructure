import os
import random
import requests
from bs4 import BeautifulSoup
import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, LiteLLMModel, Tool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# 初始化語言模型
model_id = "gemini/gemini-2.0-flash"
model = LiteLLMModel(model_id=model_id, token=os.getenv("GEMINI_API_KEY"))

# 讀取 PDF
def load_pdf(file):
    """
    從 PDF 檔案中讀取文字內容。

    Args:
        file (str): PDF 檔案路徑。

    Returns:
        tuple: 包含狀態訊息和 PDF 文字內容的元組。
    """
    file_content = ""
    try:
        pdf_reader = PdfReader(file)
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                file_content += f"\n\n===== Page {i + 1} =====\n{text}"
    except Exception as e:
        print(f"[ERROR] PDF 讀取失敗: {e}")
        return f"❌ PDF 讀取失敗：{e}", None
    if not file_content.strip():
        return "❌ PDF 無法提取任何文字內容。", None
    return "✅ PDF 讀取成功！", file_content

# 擷取 Wikipedia 條目內容（可選）
def fetch_wikipedia_text(url):
    """
    從 Wikipedia 擷取條目內容。

    Args:
        url (str): Wikipedia 條目 URL。

    Returns:
        str: 擷取的文字內容。
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find_all("p")
        return "\n".join([p.text.strip() for p in content if p.text.strip()])
    except Exception as e:
        return f"❌ 無法載入 Wikipedia 內容：{e}"

# 檢索工具
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

# 情感分析工具
def analyze_sentiment_vader(text):
    """
    使用 VADER 對給定的文本執行情感分析。

    Args:
        text (str): 要分析的輸入文本。

    Returns:
        dict: 包含 'neg'、'neu'、'pos' 和 'compound' 分數的字典。
    """
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    return vs

# AI 辯論邏輯
def start_debate(file, topic, rounds, use_theory):
    """
    啟動 AI 辯論。

    Args:
        file (str): PDF 檔案路徑。
        topic (str): 辯論主題。
        rounds (int): 辯論輪數。
        use_theory (bool): 是否使用新式奧勒岡制理論。

    Returns:
        str: 辯論紀錄。
    """
    status, content = load_pdf(file)
    if not content:
        return status

    # 取得「新式奧勒岡制」理論（可選）
    if use_theory:
        theory_text = fetch_wikipedia_text("https://zh.wikipedia.org/zh-tw/%E6%96%B0%E5%BC%8F%E5%A5%A7%E5%8B%92%E5%B2%A1%E5%88%B6")
        if theory_text.startswith("❌"):
            return theory_text
        theory_snippet = f" 理論參考（選用）：\n{theory_text[:1000]}...\n\n"
    else:
        theory_snippet = ""

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = [Document(page_content=text) for text in splitter.split_text(content)]

    tool = RetrieverTool(docs)
    agent_a = CodeAgent(tools=[tool], model=model, add_base_tools=False)
    agent_b = CodeAgent(tools=[tool], model=model, add_base_tools=False)
    roles = ["支持", "反對"]
    random.shuffle(roles)

    debate_log = f" 主題：{topic}\n Agent A 是 {roles[0]} 方，Agent B 是 {roles[1]} 方\n\n"
    statement = topic

    for i in range(rounds):
        # Agent A 發言
        related_text_a = tool.forward(query=statement)
        prompt_a = (
            f"你是辯論的{roles[0]}方。\n"
            f"請根據下方 PDF 提取的相關段落來支持此主張：{statement}\n\n"
            f" PDF 相關段落：\n{related_text_a}\n\n"
            f"{theory_snippet}"
        )
        response_a = agent_a.run(prompt_a)
        sentiment_a = analyze_sentiment_vader(response_a) #情感分析
        debate_log += f"\n Round {i+1} - Agent A（{roles[0]}）:\n{response_a}\n情感分析:{sentiment_a}\n"
        if sentiment_a['neg'] == 1.0 or sentiment_a['pos'] == 1.0:
            debate_log += f"\n 裁判：Agent A（{roles[0]}）因情感過於極端而判輸！\n"
            return debate_log

        # Agent B 反駁
        related_text_b = tool.forward(query=response_a)
        prompt_b = (
            f"你是辯論的{roles[1]}方。\n"
            f"請根據 PDF 中的內容反駁以下觀點：\n{response_a}\n\n"
            f" PDF 相關段落：\n{related_text_b}\n\n"
            f"{theory_snippet}"
        )
        response_b = agent_b.run(prompt_b)
        sentiment_b = analyze_sentiment_vader(response_b)#情感分析
        debate_log += f"\n Round {i+1} - Agent B（{roles[1]}）:\n{response_b}\n情感分析:{sentiment_b}\n"
        if sentiment_b['neg'] == 1.0 or sentiment_b['pos'] == 1.0:
            debate_log += f"\n 裁判：Agent B（{roles[1]}）因情感過於極端而判輸！\n"
            return debate_log

        statement = response_b

    debate_log += "\n 辯論結束！"
    return debate_log

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("##  PDF AI 辯論系統（基於新式奧勒岡制）")
    with gr.Row():
        pdf_file = gr.File(label="上傳 PDF 檔案", file_types=[".pdf"])
        topic_input = gr.Textbox(label="辯論主題", value="這篇文章的論證結構是否嚴謹？")
        round_slider = gr.Slider(minimum=1, maximum=5, step=1, label="辯論輪數", value=3)
        use_theory_checkbox = gr.Checkbox(label="引用新式奧勒岡制理論作為輔助", value=False)
    debate_button = gr.Button("開始辯論")
    output = gr.Textbox(label="辯論紀錄", lines=30, interactive=False)

    debate_button.click(fn=start_debate, inputs=[pdf_file, topic_input, round_slider, use_theory_checkbox], outputs=output)

# 啟動 App
if __name__ == "__main__":
    demo.launch()