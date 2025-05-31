import re
import json
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_deepseek.chat_models import ChatDeepSeek


class StoryChunkScorer:
    def __init__(self):
        # 初始化两个 LLM
        self.llm1 = ChatDeepSeek(model_name="deepseek-chat", temperature=0)
        self.llm2 = ChatDeepSeek(model_name="deepseek-chat", temperature=0)

        # 切块提示词模板
        self.split_prompt = PromptTemplate.from_template("""
你是一个数据标注助手，将用户输入的故事进行切割，组织为前后文的关系，可分割为多个具有前后文关系的块。
格式要求如下：
[
  {{"prompt": "前文", "completion": "后文"}},
  {{"prompt": "前文", "completion": "后文"}}
]
严格使用 JSON 格式返回，不要添加其他解释或描述。
故事内容如下：
{story}
""")
        self.split_chain = self.split_prompt | self.llm1

        # 评分提示词模板
        self.score_prompt = PromptTemplate.from_template("""
评价prompt和completion的上下文关联度，满分为10分，
评判规则根据completion对prompt的承接程度进行评分。
只输出数字，不作任何解释。
prompt: {prompt}
completion: {completion}
""")
        self.score_chain = self.score_prompt | self.llm2

    def extract_json_array(self, text: str):
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            return match.group(0)
        return "[]"

    def fix_model_output(self, text: str):
        text = self.extract_json_array(text)
        try:
            data = json.loads(text)
            assert isinstance(data, list)
            return data
        except Exception as e:
            print(f"❌ JSON 修复失败: {e}")
            return []

    def run_workflow(self, input_story: str) -> List[Dict]:
        try:
            split_output = self.split_chain.invoke({"story": input_story}).content
        except Exception as e:
            print(f"⚠️ LLM1（切块）生成失败：{e}")
            return []

        pairs = self.fix_model_output(split_output)
        filtered = []
        for pair in pairs:
            try:
                score = self.score_chain.invoke({"prompt": pair["prompt"], "completion": pair["completion"]}).content
                score_value = float(score)
                if score_value >= 8.0:
                    filtered.append({
                        "prompt": pair["prompt"],
                        "completion": pair["completion"],
                        "score": score_value
                    })
            except Exception as e:
                print(f"⚠️ LLM2（评分）失败，跳过当前样本：{e}")
                continue

        return filtered
