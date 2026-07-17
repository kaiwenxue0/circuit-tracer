# Circuit-tracer 使用说明

## 1. 修改配置文件

在 `circuit_tracer/configs/llama3_8B.yaml` 中进行以下修改：

* **transcoder**：设置为你的 transcoder 文件路径
* **feature\_input\_hook**：`"hook_mlp_in"`
* **feature\_output\_hook**：`"hook_mlp_out"`

---

## 2. 修改 LLaMA3 模型路径

### 2.1 在 `circuit_tracer/__main__.py`

* 找到 **第 177 行**，将其修改为 LLaMA3 模型所在的本地目录路径。

### 2.2 在 `HookedTransformer` 中

* 编辑文件：

  ```
  ./anaconda3/envs/circuit-tracer/lib/python3.10/site-packages/transformer_lens/loading_from_pretrained.py
  ```
* 将 LLaMA3 模型路径添加到 `OFFICIAL_MODEL_NAMES` 列表中。

---

## 3. 运行命令生成 graph.json 文件

在终端执行：

```bash
circuit-tracer attribute \
  --prompt "The International Advanced Security Group (IAS" \
  --transcoder_set llada1.5-8b \
  --slug kevin \
  --graph_file_dir ./graph_files_llada_date_10_30 \
  --max_feature_nodes 2000 \
  --model_type MDM
  --server
```

```bash
circuit-tracer attribute \
  --prompt "Fact: the capital of the state containing Dallas is" \
  --transcoder_set llada1.5-8b \
  --slug kevin \
  --graph_file_dir ./graph_files_llama_date_10_11 \
  --max_feature_nodes 2000
  --server
```

* **--prompt**：输入测试文本
* **--transcoder\_set**：选择使用的 transcoder 配置（此处为 `llama3`）
* **--slug**：结果标识名
* **--graph\_file\_dir**：`graph.json` 输出目录


# 前端
circuit-tracer attribute \
  --prompt "The International Advanced Security Group (IAS" \
  --transcoder_set llama3-8b \
  --slug kevin \
  --graph_file_dir ./graph_files \
  --server