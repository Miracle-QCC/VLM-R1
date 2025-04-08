cd src/open-r1-multimodal

export DEBUG_MODE="true"
export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES=0

RUN_NAME="Qwen2.5-VL-3B-GRPO-REC"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir /home/data/workgroup/qinchenjie/OmniRL/output/$RUN_NAME \
    --model_name_or_path /home/data/workgroup/qinchenjie/Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name data_config/rec.yaml \
    --image_root /home/data/workgroup/qinchenjie/datasets \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --max_output_token_length 1024
