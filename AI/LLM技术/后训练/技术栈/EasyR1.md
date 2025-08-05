# 启动 bash

```bash
#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.logger=['console','swanlab'] \
    trainer.n_gpus_per_node=8
```

- MODEL_PATH 可以是本地路径，或者直接是 huggingface 上的模型名，在训练的时候会直接从 huggingface 下载
- 主函数就是 verl.trainer.main
- data.train_files 表示 config. Yaml 里的路径，属于 data 下的 train_files
- trainer.logger=['console','swanlab'] 就是设置打印到控制台和 swanlab，就这么设置就行

# Config

分为 data、worker、algorithm、trainer 四部分

## Data


```yaml
data:
  train_files: hiyouga/math12k@train
  val_files: hiyouga/math12k@test
  prompt_key: problem
  answer_key: answer
  image_key: images
  video_key: videos
  image_dir: null
  video_fps: 2.0
  max_prompt_length: 2048
  max_response_length: 2048
  rollout_batch_size: 512  # equivalent to verl's data.train_batch_size
  mini_rollout_batch_size: null  # equivalent to verl's data.gen_batch_size
  val_batch_size: 1024
  format_prompt: ./examples/format_prompt/math.jinja
  override_chat_template: null
  shuffle: true
  seed: 1
  min_pixels: 262144
  max_pixels: 4194304
  filter_overlong_prompts: true
```

| 参数名称                    | 解读                                                                                                                                                                                                                          |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| train_files             | 训练数据集文件路径。@后train 或者 val ，区分训练集和验证集。由于加载数据集时使用[datasets.load_dataset](https://zhida.zhihu.com/search?content_id=259635580&content_type=Article&match_order=1&q=datasets.load_dataset&zhida_source=entity)，可以支持多种类型的文件作为数据集。 |
| val_files               | 验证数据集文件路径。                                                                                                                                                                                                                  |
| prompt_key              | 每条数据中prompt对应的key。                                                                                                                                                                                                          |
| answer_key              | 每条数据中label对应的key。                                                                                                                                                                                                           |
| image_key               | 每条数据中image对应的key。                                                                                                                                                                                                           |
| image_dir               | 图片所在的目录路径。若不为None，则会将image_dir和数据中的图像路径拼接成新的图像路径。                                                                                                                                                                           |
| max_prompt_length       | prompt的最大长度限制。                                                                                                                                                                                                              |
| max_response_length     | response的最大长度限制                                                                                                                                                                                                             |
| rollout_batch_size      | Batch size sampled for one training iteration of different RL algorithms. 每次梯度更新真正要用的 有效 prompt 条数（全局视角）。在代码中，对应train_data_loader的batch size                                                                                |
| mini_rollout_batch_size | 每次调用推理引擎一次性喂进去的 prompt 数（可多轮生成、可 > train_batch_size）。在代码中，当传入mini_rollout_batch_size时，作为 train_data_loader的batch size，不再使用 rollout_batch_size。                                                                              |
| val_batch_size          | 验证时的batch size                                                                                                                                                                                                              |
| format_prompt           | jinjia prompt模板路径。每条数据的prompt会和这个模板结合，render成新的prompt。                                                                                                                                                                      |
| override_chat_template  | 传入chat template时覆盖掉默认的chat template。例如：与Qwen2.5VL在同一目录下的chat_template.json。                                                                                                                                                 |
| shuffle                 | 是否shuffle训练数据。                                                                                                                                                                                                              |
| seed                    | shuffle训练数据传入的random seed。                                                                                                                                                                                                  |
| min_pixels              | Qwen2/2.5-VL系列图像预处理时，对图像的最低pixel要求。                                                                                                                                                                                         |
| max_pixels              | Qwen2/2.5-VL系列图像预处理时，对图像的最高pixel要求。                                                                                                                                                                                         |
| filter_overlong_prompts | 是否将大于max_prompt_length的prompt从训练数据集中过滤出去。                                                                                                                                                                                   |
## Algorithm

```yaml
algorithm:
  adv_estimator: grpo
  disable_kl: false
  use_kl_loss: true
  kl_penalty: low_var_kl
  kl_coef: 1.0e-2
  online_filtering: false  # dapo filter groups
  filter_key: overall
  filter_low: 0.01
  filter_high: 0.99
```

训练多模态的时候可以将 disable_kl 设置为 true

## Worker

