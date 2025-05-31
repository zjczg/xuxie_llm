import json

# 输入和输出文件路径
input_file = "my_data.jsonl"
output_file = "my_data_alpaca.jsonl"

# 固定instruction提示词
instruction = "续写以下小说片段："

# 计数器
count = 0

with open(input_file, "r", encoding="utf-8") as f_in, \
        open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        # 解析原始数据
        data = json.loads(line)
        prompt = data["prompt"]
        completion = data["completion"]

        # 转换为Alpaca格式
        alpaca_data = {
            "instruction": instruction,  # 固定提示词
            "input": prompt,  # 原始prompt作为输入
            "output": completion  # 原始completion作为输出
        }

        # 写入新文件
        f_out.write(json.dumps(alpaca_data, ensure_ascii=False) + "\n")
        count += 1

print(f"转换完成！共处理 {count} 条数据")
print(f"输出文件: {output_file}")