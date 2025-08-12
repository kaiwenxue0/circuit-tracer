# Circuit-tracer的使用：

1.修改config文件：
在'circuit_tracer/configs/llama3_8B.yaml'文件中，修改transcoder的地址，以及feature_input_hook: "hook_mlp_in", feature_output_hook: "hook_mlp_out".

2.修改llama3模型地址：
2.1 在'circuit_tracer/__main__.py'文件中，第177行，修改llama3模型所在目录；
2.2 在'HookedTransformer'中，记得把llama3模型目录添加到'OFFICIAL_MODEL_NAMES'中。'./anaconda3/envs/circuit-tracer/lib/python3.10/site-packages/transformer_lens/loading_from_pretrained.py'

3.运行命令, 得到对应的graph.json文件。
circuit-tracer attribute \
  --prompt "The International Advanced Security Group (IAS" \
  --transcoder_set llama3 \
  --slug [Your_slug] \
  --graph_file_dir ./graph_files
