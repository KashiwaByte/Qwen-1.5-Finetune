# 任务介绍
通过在基座大模型上进行指令微调的方式实现文本分类任务


数据集为jsonl格式，每行json由 给定文本（Text），文本类型选项（Catagory），正确文本类型（output）,三个部分组成

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240602214420.png)

# 具体步骤


## 数据集处理

### 数据集下载
我们首先通过ModelScope的方式，加载给定的数据集
```
from modelscope import MsDataset
dataset = MsDataset.load('huangjintao/zh_cls_fudan-news', split='train')
test_dataset = MsDataset.load('huangjintao/zh_cls_fudan-news', subset_name='test', split='test')
print(dataset)
print(test_dataset)
```

### 指令微调数据集构造
我们需要编写一段Python脚本，将原有的数据集构造成<instruction,input,output>这样三元组格式的指令微调数据集，

- 在instruction中我们需要通过提示词工程的方法编写一段提示词
- 接着我们需要将原本数据集中的给定文本（Text），文本类型选项（Catagory)都放到input中
- 最后我们需要将原本的正确文本类型（output），放到output中
- 然后我们将生成的三元组格式的数据集统一写入一个新的jsonl文件中

```
import csv
import json


jsonl_file = 'news_train.jsonl'

# 生成JSONL文件
messages = []


# 读取jsonl文件
with open(dataset, 'r') as file:
    for line in file:
        # 解析每一行的json数据
        data = json.loads(line)
        context = data["text"]
        catagory = data["category"]
        label = data["output"]
        message={ "instruction":"你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型","input": f'文本:{context},类型选型:{catagory}',"output":label}
        messages.append(message)

# 保存为JSONL文件
with open(jsonl_file, 'w', encoding='utf-8') as file:
    for message in messages:
        file.write(json.dumps(message, ensure_ascii=False) + '\n')
```



### 数据集加载
接下来我们需要数据集加载为CSV格式，用于模型读取
```
from datasets import Dataset
import pandas as pd

# 将jsonl文件转换为CSV文件
df = pd.read_json('./news_train.jsonl',lines = True)
ds = Dataset.from_pandas(df)
```


### 数据格式化
`Lora` 训练的数据是需要经过格式化、编码之后再输入给模型进行训练的，如果是熟悉 `Pytorch` 模型训练流程的同学会知道，我们一般需要将输入文本编码为 input_ids，将输出文本编码为 `labels`，编码之后的结果都是多维的向量。我们首先定义一个预处理函数，这个函数用于对每一个样本，编码其输入、输出文本并返回一个编码后的字典：

```
def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\n你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

```
在构造的时候需要根据不同的模型提示词模板进行相应的调整，Qwen1.5的提示词模版如下

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你是谁？<|im_end|>
<|im_start|>assistant
我是一个有用的助手。<|im_end|>
```


接下来我们通过通过刚刚的模型格式化函数，将我们的数据集转化为tokenized_id数据集用于后续训练使用
```
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
```
## 模型加载

###  加载tokenizer和半精度模型
模型以半精度形式加载，如果你的显卡比较新的话，可以用`torch.bfolat`形式加载。对于自定义的模型一定要指定`trust_remote_code`参数为`True`。

```
tokenizer = AutoTokenizer.from_pretrained('./qwen/Qwen1.5-7B-Chat/', use_fast=False, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained('./qwen/Qwen1.5-7B-Chat/', device_map="auto",torch_dtype=torch.bfloat16)

model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
```

### 设置LoRA参数
`LoraConfig`这个类中可以设置很多参数，具体深入可以看源码

- `task_type`：模型类型
- `target_modules`：需要训练的模型层的名字，主要就是`attention`部分的层，不同的模型对应的层的名字不同，可以传入数组，也可以字符串，也可以正则表达式。
- `r`：`lora`的秩，具体可以看`Lora`原理
- `lora_alpha`：`Lora alaph`，具体作用参见 `Lora` 原理

`Lora`的缩放是啥嘞？当然不是`r`（秩），这个缩放就是`lora_alpha/r`, 在这个`LoraConfig`中缩放就是4倍。


```
from peft import LoraConfig, TaskType, get_peft_model

  

config = LoraConfig(

    task_type=TaskType.CAUSAL_LM,

    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    inference_mode=False, # 训练模式

    r=8, # Lora 秩

    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理

    lora_dropout=0.1# Dropout 比例

)

config
```


### 加载LoRA参数
接下来我们借助peft库中的方法，将我们前面加载的模型与LoRA 设置合并为一个最终的模型

```
model = get_peft_model(model, config)
```


## 模型训练

### 配置训练参数
`TrainingArguments`这个类的源码也介绍了每个参数的具体作用，当然大家可以来自行探索，这里就简单说几个常用的。

- `output_dir`：模型的输出路径
- `per_device_train_batch_size`：顾名思义 `batch_size`
- `gradient_accumulation_steps`: 梯度累加，如果你的显存比较小，那可以把 `batch_size` 设置小一点，梯度累加增大一些。
- `logging_steps`：多少步，输出一次`log`
- `num_train_epochs`：顾名思义 `epoch`
- `gradient_checkpointing`：梯度检查，这个一旦开启，模型就必须执行

```
args = TrainingArguments(
    output_dir="./output/Qwen1.5",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)
```


### 利用SwanLab实现模型监控
在配置完参数后我们推荐使用模型实验管理工具来记录实验的训练情况，这样我们就不需要待在命令行面前盯着打印的结果了。

[SwanLab]([SwanLab - 云上的AI实验平台，一站式跟踪、比较、分享你的模型，一站式AI实验协作，跟踪超参数和指标，监控GPU与硬件情况](https://swanlab.cn/))是一个高效，好用的模型实验管理python库，可以很方便地对我们的训练任务进行记录，并提供可视化的分析图表.
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240602221034.png)

SwanLab的使用流程也非常简单，只需以下三步即可：

首先我们需要通过以下命令下载SwanLab库
```
pip install swanlab
```

然后我们需要去SwanLab的官方网站注册一个账号（通过手机注册即可，非常方便）

接着进入我们的个人设置界面**获取 API-Key**（这是将我们的数据上传到云端的关键）
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240602222022.png)


目前SwanLab已经对十几个主流的开源框架进行了集成，让我们可以通过几行代码轻松实现日志记录。我们通过查询它的官方文档可以发现，SwanLab目前已经支持了Transformers  Trainer的支持。

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240602221222.png)





### 设置Trainer开启训练
不得不说SwanLab的官方文档写的非常详尽，我们只需要参考示例教程，在Trainer中传入callbacks回调参数即可轻松记录实验。

```
from swanlab.integration.huggingface import SwanLabCallback

swanlab_callback = SwanLabCallback(project="hf-visualization")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)


trainer.train()
```

在我们开启的训练之后，SwanLab会自动要求填入API-key，填入我们在上一个环节获取的API-key即可。

填写完毕后，训练就将开启。


### 查看训练进展
SwanLab目前支持了在Jupyter界面中直接开启看板，来很方便的了解训练的情况。

![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240602222728.png)

当然也可以直接登录官网，在我们的个人账号下进行查看。
![image.png](https://kashiwa-pic.oss-cn-beijing.aliyuncs.com/20240602222840.png)


## 模型推理
训练好了之后可以使用如下方式加载`LoRA`权重进行推理：

```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = './qwen/Qwen1.5-7B-Chat/'
lora_path = 'lora_path'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

prompt = "(测试集文本与选项)"
messages = [
    {"role": "system", "content": "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
```



```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

mode_path = './qwen/Qwen1.5-7B-Chat/'
lora_path = 'output/Qwen1.5/checkpoint-600'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)


import csv
import json


jsonl_file = 'zh_cls_fudan-news/test.jsonl'
result = 'result.jsonl'
# 生成JSONL文件
messages = []


# 读取jsonl文件
with open(jsonl_file, 'r') as file:
    for line in file:
        # 解析每一行的json数据
        data = json.loads(line)
        context = data["text"]
        catagory = data["category"]
        prompt = f'文本:{context},类型选型:{catagory}'
        messages = [
            {"role": "system", "content": "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型"},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        message = response
        print(response)
        messages.append(message)

# 保存为JSONL文件
with open(result, 'w', encoding='utf-8') as file:
    for message in messages:
        file.write(json.dumps(message, ensure_ascii=False) + '\n')

```