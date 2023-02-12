import torch
import numpy as np
from datasets import load_metric
from transformers import ViTForImageClassification, Trainer, ViTImageProcessor
from args import training_args

metric = load_metric("accuracy")
def_model_path = "google/vit-large-patch16-224"
feature_extractor = ViTImageProcessor(def_model_path)
num_labels = 2

# for preprocess
def transform(example_batch):
    """
    dynamic transform function for batches
    """
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
  """
  metric function called training / eval
  """
  return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def load_model(dataset, model_path=def_model_path):
  """
  return trainer for fine-tuning model
  """
  model = ViTForImageClassification.from_pretrained(
    model_path,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
  )
  
  trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=feature_extractor,
  )

  return trainer

