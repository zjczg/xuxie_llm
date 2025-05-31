# Lora微调qwen1.8B总结

## 数据集准备：

### 数据格式：

#### 指令微调格式Alpaca

```json
[
  {
    "instruction": "解释量子力学的基本概念",
    "input": "",
    "output": "量子力学是描述微观粒子行为的物理学理论...",
    "system": "你是一个物理学专家",
    "history": [
      ["第一轮问题", "第一轮回答"],
      ["第二轮问题", "第二轮回答"]
    ]
  }
]
```

datset_info.json的配置

```json
"my_data": {
  "file_name": "data.json",
  "formatting": "alpaca",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

#### 我的数据格式：

```json
"my_data": {
  "file_name": "my_data.jsonl",
  "formatting": "alpaca",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "",
    "history": ""
  }
}
```

```json
{"instruction": "续写以下小说片段：", "input": "二愣子睁大着双眼，直直望着茅草和烂泥糊成的黑屋顶，身上盖着的旧棉被，已呈深黄色，看不出原来的本来面目，还若有若无的散发着淡淡的霉味。", "output": "在他身边紧挨着的另一人，是二哥韩铸，酣睡的十分香甜，从他身上不时传来轻重不一的阵阵打呼声。"}

```

input为上文，output为下文

原未转化数据：

```json
{"prompt": "二愣子睁大着双眼，直直望着茅草和烂泥糊成的黑屋顶，身上盖着的旧棉被，已呈深黄色，看不出原来的本来面目，还若有若无的散发着淡淡的霉味。", "completion": "在他身边紧挨着的另一人，是二哥韩铸，酣睡的十分香甜，从他身上不时传来轻重不一的阵阵打呼声。"}

```

### 数据集生成：

generate_data两个LLM，一个分析chunck生成数据，一个评分。

get_data划分chunck调用generate_data将符合条件的训练数据转化为jsonl。

(被gpt骗了，get_data生成的数据格式llamafactory不接受，最后加一个convert_data转化数据格式为Alpaca)

## 环境配置：

### 模型下载

git lfs下载模型或者hugging face的python和魔塔的包都可以下载模型，但llamafactory关于模型的适配是依照hugging face的(最好下hugging face的模型)。

### llamafactory的安装

```
git clone https://github.com/hiyouga/LLaMA-Factory.git
pip install -U pip
pip install -e ".[torch,metrics]"
```

### 模型的环境安装

```
pip install transformers==4.50.1 peft accelerate bitsandbytes datasets
```

> [!NOTE]
>
> 注意transformer的版本模型所依赖的版本会和llamafactory的产生冲突。

## 训练

```
CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --stage sft \
  --do_train \
  --model_name_or_path /root/autodl-fs/models/Qwen-1_8B \
  --dataset my_data \
  --dataset_dir ./data \
  --finetuning_type lora \
  --output_dir ./saves/Qwen-1_8B/lora/my_finetune \
  --template alpaca \
  --lora_target all \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --lr_scheduler_type cosine \
  --fp16 \
  --logging_steps 10 \
  --save_steps 200 \
  --overwrite_cache \
  --plot_loss \
  --trust_remote_code

```

## 推理

```
llamafactory-cli chat \
  --model_name_or_path /root/autodl-fs/models/Qwen-1_8B \
  --adapter_name_or_path ./saves/Qwen-1_8B/lora/my_finetune \
  --template qwen \
  --finetuning_type lora \
  --trust_remote_code \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 50 \
  --repetition_penalty 1.2 \
  --max_new_tokens 300
  --system "你是一个严谨的故事续写专家，为用户的输入进行续写"

```

## Tip:

generate方法：

llamafactory的patch文件中的

```
model.generate = MethodType(PreTrainedModel.generate, model)
```

是transformer的GenerationMixin类自带generate方法会自动给模型加上transformer自带的generate方法，但是qwen有自己的生成逻辑重写generate方法在modeling_qwen文件中，我使用trust_remote_code加载本地的模型，就是使用qwen自己的generate方法，这里使用transformer的会出错，且低版本的transformer不带generate方法。

修改：

```
if hasattr(model, "generate"):
    pass  # 模型自己有 generate，就不动它
else:
    model.generate = MethodType(PreTrainedModel.generate, model)
```

## 结果：

### 微调后的：

![image-20250531190756051](C:\Users\15143\AppData\Roaming\Typora\typora-user-images\image-20250531190756051.png)

### Ollamam本地的：

![image-20250531190831325](C:\Users\15143\AppData\Roaming\Typora\typora-user-images\image-20250531190831325.png)

### 结论：

我太菜了1.8B是真的玩不起来，不像14B听得懂人话。

微调后的模型在前端的续写上扩展了人物，但在后段全是乱编（初步怀疑是喂的数据没有长文，乱编的数据是没有相应的知识编不下去了，就乱生成)

本地的模型基本就在原来内容上进行不痛不痒的扩展。

总的来说知识是学了，但只能学一点点。

总结，一个数据集的问题，可能缺乏长文续写的数据集，再一个模型需要换更大的(优化目标)