from transformers import TrainingArguments

output_dir = "vit-large-ai-or-not"

training_args = TrainingArguments(
  output_dir=output_dir,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=2,
  save_steps=200,
  fp16=True,
  eval_steps=200,
  logging_steps=200,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=True,
  load_best_model_at_end=True,
)