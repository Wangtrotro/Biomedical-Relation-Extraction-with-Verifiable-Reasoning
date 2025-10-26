# ============================================
# Biomedical Relation Extraction with Verifiable Reasoning
# Compatible with: LangChain 0.3+, BioGPT local model
# ============================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # ✅ 优先当前目录

from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
import pandas as pd
import json
from relation_verifier import verify_relation


# ======================
# 1️⃣ 加载 BioGPT 模型
# ======================
print("🚀 Loading BioGPT model from local cache...")
generator = pipeline(
    "text-generation",
    model="/Users/shihansmac/.cache/huggingface/hub/models--microsoft--BioGPT/snapshots/eb0d815e95434dc9e3b78f464e52b899bee7d923",
    tokenizer="microsoft/biogpt",
    max_new_tokens=256,
    temperature=0.3,
    pad_token_id=50256
)
llm = HuggingFacePipeline(pipeline=generator)
print("✅ BioGPT model loaded successfully.\n")


# ======================
# 2️⃣ 定义 Prompt 模板
# ======================
template = """
任务：从以下生物医学文本中抽取关系并解释推理过程。
输出格式必须为 JSON，包含以下字段：
{{
  "head": "...",
  "relation": "...",
  "tail": "...",
  "evidence": "...",
  "reasoning_trace": ["Step1: ...", "Step2: ..."]
}}
文本：{input_text}
输出：
"""
prompt = PromptTemplate.from_template(template)


# ======================
# 3️⃣ 定义轻量链式类 (替代 LLMChain)
# ======================
class SimpleChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_text):
        text_prompt = self.prompt.format(input_text=input_text)
        result = self.llm.invoke(text_prompt)
        return result


# ======================
# 4️⃣ 载入知识库
# ======================
data_path = os.path.join(os.path.dirname(__file__), "data", "biokg.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ 未找到知识库文件: {data_path}")
kg = pd.read_csv(data_path)
print(f"📚 Loaded knowledge base from {data_path} ({len(kg)} entries)\n")


# ======================
# 5️⃣ 实例化链式处理
# ======================
bio_chain = SimpleChain(llm=llm, prompt=prompt)


# ======================
# 6️⃣ 运行任务示例
# ======================
if __name__ == "__main__":
    input_text = "Aspirin reduces inflammation by inhibiting COX-2 enzyme."
    print(f"🧬 输入文本：{input_text}\n")

    result = bio_chain.run(input_text=input_text)
    print("🧠 模型输出：\n", result)

    verification_result = verify_relation(result, kg)
    print("\n🔍 验证结果：\n", verification_result)

    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), "results", "demo_output.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "input": input_text,
            "model_output": result,
            "verification": verification_result
        }, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存至: {output_path}\n")
