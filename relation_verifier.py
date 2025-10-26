import json
import pandas as pd

def verify_relation(output_text, kg: pd.DataFrame):
    """
    验证 BioGPT 输出的三元组是否与知识库一致。
    """
    try:
        s, e = output_text.find("{"), output_text.rfind("}")
        if s == -1 or e == -1:
            return "❌ 未检测到 JSON 结构。"
        data = json.loads(output_text[s:e+1])
    except Exception as e:
        return f"❌ JSON 解析错误：{e}"

    # 验证结构字段
    required = ["head", "relation", "tail", "evidence"]
    if not all(k in data for k in required):
        return "⚠️ 缺少必要字段。"

    # 验证知识库匹配
    hit = ((kg["head"] == data["head"]) &
           (kg["relation"] == data["relation"].upper()) &
           (kg["tail"] == data["tail"])).any()

    if hit:
        return f"✅ 验证通过：({data['head']} --{data['relation']}--> {data['tail']})"
    else:
        return f"⚠️ 未在知识库中找到匹配关系 ({data['head']} --{data['relation']}--> {data['tail']})。"
