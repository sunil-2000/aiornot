import torch
import numpy as np
from datasets import load_metric
from transformers import ViTForImageClassification, TrainingArguments, Trainer, ViTImageProcessor

metric = load_metric("accuracy")
def_model_path = "google/vit-base-patch16-224"
feature_extractor = ViTImageProcessor(def_model_path)
num_labels = 2

training_args = TrainingArguments(
  output_dir="./vit-base-ai-or-not",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=1,
  save_steps=200,
  fp16=True,
  eval_steps=200,
  logging_steps=200,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  load_best_model_at_end=True,
)
test_args = TrainingArguments(
  output_dir="./vit-base-ai-or-not",
  do_train=False,
  do_predict=True,
  per_device_eval_batch_size=16,  
  remove_unused_columns=False,
  fp16=True, 
)

# for preprocess
def transform(example_batch):
    inputs = feature_extractor(
        [x for x in example_batch["image"]], return_tensors="pt"
    )
    inputs['id'] = example_batch['id']
    inputs["labels"] = example_batch["label"]
    return inputs

# for train
def collate_fn(batch):
  """
  data collator function
  """
  return {
    'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
    'labels': torch.tensor([x['labels'] for x in batch])
  }

def compute_metrics(p):
  return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def load_model(dataset=None, model_path=def_model_path):
  model = ViTForImageClassification.from_pretrained(
    model_path,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
  )
  if dataset:
    trainer = Trainer(
      model=model,
      args=training_args,
      data_collator=collate_fn,
      compute_metrics=compute_metrics,
      train_dataset=dataset["train"],
      eval_dataset=dataset["test"],
      tokenizer=feature_extractor,
    )
  else:
      trainer = Trainer(
      model=model,
      args=test_args,
      data_collator=collate_fn,
      tokenizer=feature_extractor,
    )
  return trainer

