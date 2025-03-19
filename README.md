# OmniRL: Omni Reward and Loss Scheme for Vision-Language R1 Model Training
Due to the success of DeepSeek R1, many researchers have been drawn into the reproduction of R1. However, when attempting to directly use the open-source multimodal R1 framework, various issues are often encountered. Therefore, we have developed a framework called OmniRL based on [VLM-R1](https://github.com/om-ai-lab/VLM-R1) that supports training in pure text, text-image, multi-image, and mixed text-only and image-text modes.

GRPO utilizes only two simple rule-based reward functions: format reward and accuracy reward. The implementation of accuracy reward varies by task. For tasks with fixed-answer selection, it is sufficient to determine whether the model's output matches the ground truth. For LeetCode problems, a compiler is necessary to assess correctness. However, for the majority of generative tasks, the quality of model-generated outputs can be evaluated using three metrics: BLEU, ROUGE, and METEOR. Based on this, we propose using these metrics to directly calculate local matching degree (Rb), critical information coverage (Rr), and semantic coherence (Rm) as reward functions, forming a unified accuracy reward function applicable to reinforcement learning across various tasks.

## Features
- Supported inputs
  - text-only
  - image-text pair
  - multi-images
  - mixed text-only and image-text in one json

- Supported datasets
  - text-only format: 
    - [{"problem": question, "solution": answer}, ……]
  - image-text pair: 
    - [{"problem": question, "solution": answer, "image": [image_path]}, ……]
  - mixed text-only and image-text in one json:
    - [{"problem": question, "solution": answer}, {"problem": question, "solution": answer, "image": [image_path]}, ……]
  
- Reward functions
  - We utilize generative evaluation metrics such as BLEU and ROUGE as accuracy reward functions, which can be adapted to most tasks.

## Training Skills
- The reward functions primarily consist of three types: length reward, accuracy reward, and format reward. The values of these three reward functions should not differ too much. For example, if the length reward is greater than 500 while the accuracy reward is between 0 and 1, the convergence of accuracy would be much slower. 
- Learning the format is quite straightforward and converges quickly, so it doesn't require much effort. 
- We have set a parameter called "max_output_token_length" to indicate the maximum number of tokens the model can output. When the response is too long, this parameter needs to be set larger, otherwise the response will be truncated, causing the format reward to remain at 0.

## Our Teams
- We are the Multimodal Algorithm Team of AIDC. If you are looking for a job, please feel free to contact us: yilei.yi@alibaba-inc.com

## Setup

```bash
conda create -n vlm-r1 python=3.10
conda activate vlm-r1
bash setup.sh
```

## Training

### Referring Expression Comprehension (REC)

> 1. Download the [COCO Train2014 image](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/train2014.zip) and unzip it, and we refer to the image dir as `<your_image_root>`.

> 2. Download the [RefCOCO/+/g Annotation files](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip) and unzip it.

> 3. Write the path of the annotation files in the `src/open-r1-multimodal/data_config/rec.yaml` file.
```bash
datasets:
    - json_path: /path/to/refcoco_train.json
    - json_path: /path/to/refcocop_train.json
    - json_path: /path/to/refcocog_train.json
```

> 4. ```bash src/open-r1-multimodal/run_grpo_rec.sh```

```bash
cd src/open-r1-multimodal

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name data_config/rec.yaml \
    --image_root <your_image_root> \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true
```

> 1. Download the provided [RefGTA images](https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/refgta.zip).
```bash
cd ./src/eval

# Remember to change the model path, image root, and annotation path in the script
python test_rec_r1.py 
```

## Acknowledgements

We would like to express our sincere gratitude to [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [R1-V](https://github.com/Deep-Agent/R1-V), [RefCOCO](https://github.com/lichengunc/refer), and [RefGTA](https://github.com/mikittt/easy-to-understand-REG/tree/master/pyutils/refer2) for providing open-source resources that contributed to the development of this project.



## Citation
If you find this project useful, welcome to cite us.
```bib
@misc{shen2025vlmr1,
  author       = {Shen, Haozhan and Zhang, Zilun and Zhang, Qianqian and Xu, Ruochen and Zhao, Tiancheng},
  title        = {VLM-R1: A stable and generalizable R1-style Large Vision-Language Model},
  howpublished = {\url{https://github.com/om-ai-lab/VLM-R1}},
  note         = {Accessed: 2025-02-15},
  year         = {2025}
}
```

```bib
@misc{long2025OmniRL,
  author       = {Long, Rujiao and Jin, Ziyu and Wang, Zhan and Huang, Zijin and Cheng, Qiannan and Yi, Lei},
  title        = {OmniRL: Omni Reward and Loss Scheme for Vision-Language R1 Model Training},
  howpublished = {\url{https://github.com/alibaba/OmniRL}},
  note         = {Accessed: 2025-03-18},
  year         = {2025}
}
```
