from smolagents import CodeAgent, HfApiModel

model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

model = HfApiModel(model_id=model_id, token="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")
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you tell me 1+1 ?",
)

