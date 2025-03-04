from smolagents import CodeAgent, HfApiModel

model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

model = HfApiModel(model_id=model_id, token="your huggingface  token")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you tell me 1+1 ?",
)

