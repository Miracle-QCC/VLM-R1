# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
import torch.distributed as dist

from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import Levenshtein as lev

from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
from trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "length"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    max_output_token_length: Optional[int] = field(
        default=128,  # 默认值
        metadata={"help": "生成文本的最大长度"}
    )

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

        # 新增分类索引
        self.image_indices = []
        self.text_indices = []
        for idx in range(len(self.list_data_dict)):
            if 'image' in self.list_data_dict[idx] and len(self.list_data_dict[idx]["image"]) > 0:
                self.image_indices.append(idx)
            elif 'image' in self.list_data_dict[idx]:
                del self.list_data_dict[idx]['image']
                self.text_indices.append(idx)
            else:
                self.text_indices.append(idx)

    # 新增方法获取分类信息
    def get_indices(self):
        return self.image_indices, self.text_indices

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }
        # FIXME
        # This is only for Grounding task
        QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
        def make_conversation_image(example):
            return {
                "prompt": [
                    # {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                        ],
                    },
                ],
            }

        has_image = 0
        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if 'image' in example:
            if isinstance(example['image'], list):
                image_path = [os.path.join(image_root, x) for x in example['image']]
            else:
                image_path = [os.path.join(image_root, example['image'])]
            
            images = []
            for path in image_path:
                try:
                    image = Image.open(path).convert("RGB")
                    width, height = image.size
                    images.append(image)
                except Exception as e:
                    print("read image err: " + path)
                    continue
                if width < 50 or height < 50:
                    continue
            if len(images) == 0:
                images.append(Image.new('RGB', (224, 224), (255, 255, 255)))
            has_image = 1

            return {
                'has_image': has_image,
                'image': images,
                'problem': example['problem'],
                'solution': example['solution'],
                'prompt': make_conversation_image(example)['prompt']
            }
        
        return {
            'has_image': has_image,
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation(example)['prompt'],
        }

from torch.utils.data import DataLoader

class BalancedBatchSampler:
    def __init__(self, image_indices, text_indices, batch_size):
        self.batch_size = batch_size
        self.image_indices = image_indices.copy()
        self.text_indices = text_indices.copy()

    def __iter__(self):
        # 合并所有可能的batch
        all_batches = []
        # 图像batch
        random.shuffle(self.image_indices)
        for i in range(0, len(self.image_indices), self.batch_size):
            all_batches.append(self.image_indices[i:i+self.batch_size])
        # 文本batch
        random.shuffle(self.text_indices)
        for i in range(0, len(self.text_indices), self.batch_size):
            all_batches.append(self.text_indices[i:i+self.batch_size])
        # 随机打乱batch顺序
        random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        return (len(self.image_indices) + self.batch_size - 1) // self.batch_size + \
               (len(self.text_indices) + self.batch_size - 1) // self.batch_size

def collate_fn(batch):
    # Format into conversation
    return batch

'''
    If the iou of the bbox predicted by the model and the ground truth is greater than 0.5, the reward is 1.0, otherwise 0.0 .
    This is a hard reward, maybe the soft reward is better and could be used in the future .
'''
def iou_reward(completions, solution, **kwargs):
    def iou(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2]-1, box2[2]-1)
        inter_y2 = min(box1[3]-1, box2[3]-1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
        return float(inter)/union
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*),\s*(\s*-?\d*\.?\d+\s*)\]'
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()
                bbox_match = re.search(bbox_pattern, content_answer)
                if bbox_match:
                    bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                    if iou(bbox, sol) > 0.5:
                        reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r'<think>.*?</think>.*?<answer>.*?</answer>'
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def len_reward(completions, solution, L=500, gamma=0.9, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    lengths = [len(content) for content in contents]

    rewards = []
    punctuation_counts = [sum([1 for c in text if c in ',.;!']) for text in contents]
    for length, punctuation_count in zip(lengths, punctuation_counts):
        density = punctuation_count / (length + 1e-6)

        dynamic_alpha = gamma ** (length//L)

        if length <= L:
            base = length * (1 + math.sin(math.pi*length/(2*L)))
        else:
            base = 2*L + math.log(length-L+1)

        reward = base * dynamic_alpha * (0.5 + 0.5*density)
        rewards.append(reward)

    return [x / 500 for x in rewards]

# def accuracy_reward(completions, solution, **kwargs):
#     pattern = r'<answer>(.*?)</answer>'
#     contents = [completion[0]["content"] for completion in completions]
#     if dist.get_rank() == 0:
#         print("\n******************************************************response***************************************************\n", contents[0], "\n******************************************************response***************************************************\n".replace("*", "-"))
#     rewards = []
#     for content, sol in zip(contents, solution):
#         reward = 0.0
#         content_answer_match = re.search(pattern, content, re.DOTALL)
#         if content_answer_match:
#             content_answer = content_answer_match.group(1).strip()
#             if content_answer == sol:
#                 reward = 1.0
#         rewards.append(reward)

#     return rewards

def accuracy_reward(completions, solution, **kwargs):
    pattern = r'<answer>(.*?)</answer>'
    contents = [completion[0]["content"] for completion in completions]
    if dist.get_rank() == 0:
        print("\n******************************************************response***************************************************\n", contents[0], "\n******************************************************response***************************************************\n".replace("*", "-"))
    rewards = []
    for content, sol in zip(contents, solution):
        reward = 0.0
        content_answer_match = re.search(pattern, content, re.DOTALL)
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            reward = (bleu(content_answer, sol) + rouge(content_answer, sol)) / 2
        rewards.append(reward)

    return rewards

def bleu(prediction, reference):
    prediction = re.sub(r'[^\sa-zA-Z0-9]', '', prediction)
    reference = re.sub(r'[^\sa-zA-Z0-9]', '', reference)
    reference = [reference.split()]
    # 机器翻译结果
    prediction = prediction.split()
    bleu_score = sentence_bleu(reference, prediction, weights=(0.5,0.5))
    return bleu_score

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeL'], use_stemmer=True)
def rouge(prediction, reference):
    score = scorer.score(reference,
                        prediction)
    return score['rouge1'].fmeasure

def rougeL(prediction, reference):
    score = scorer.score(reference,
                        prediction)
    return score['rougeL'].fmeasure

def edit_similarity_score(predictions, references):
    dist = lev.distance(predictions, references)
    max_len = max(len(predictions), len(references))
    return 1 - dist / max_len if max_len != 0 else 1

reward_funcs_registry = {
    # "accuracy": iou_reward,
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": len_reward,
}


def main(script_args, training_args, model_args):
    training_args.max_completion_length = script_args.max_output_token_length
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args)

    # 创建平衡采样器
    image_indices, text_indices = dataset.get_indices()
    batch_sampler = BalancedBatchSampler(
        image_indices=image_indices,
        text_indices=text_indices,
        batch_size=training_args.per_device_train_batch_size
    )

    # 创建标准DataLoader
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,  # 需自定义collate函数
        num_workers=training_args.dataloader_num_workers
    )

    trainer_cls = Qwen2VLGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        data_collator=collate_fn,
        batch_sampler=batch_sampler,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
