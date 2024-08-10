cd ..
accelerate launch --config_file scripts/accelerate_config.json run.py \
	--train_file "0.5 data/train/train_plan_gen.json 0.5 data/train/train_act_recog.json 0.5 data/train/train_count.json 0.5 data/train/train_obj_move.json" \
	--fisher_matrix_path fisher-matrix/fisher-matrix-6B \
	--model_name_or_path EleutherAI/gpt-j-6B \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 2 \
	--lr 8e-5 \
	--output_dir output/ewc-lora-6B/checkpoint \
	--num_epochs 5 \
	--ewc_lambda 0.5