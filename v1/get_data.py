import os
import json
from dotenv import load_dotenv
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from generate_data import StoryChunkScorer

def main():
    # 加载 .env 中的环境变量
    load_dotenv()
    os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")

    scorer = StoryChunkScorer()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]
    )

    data_path = './doc/'
    output_path = 'finetune_dataset.jsonl'

    all_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for filename in tqdm(all_files, desc="📄 处理文件"):
            file_path = os.path.join(data_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = splitter.split_text(text)

                for chunk in tqdm(chunks, leave=False, desc=f"🔍 {filename}"):
                    result_list = scorer.run_workflow(chunk)
                    if not result_list:
                        continue

                    for item in result_list:
                        try:
                            record = {
                                "prompt": item["prompt"].strip(),
                                "completion": item["completion"].strip()
                            }
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        except Exception as e:
                            print(f"⚠️ 写入失败：{e}")

if __name__ == "__main__":
    main()
