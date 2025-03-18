from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, pipeline
import evaluate
import numpy as np
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=False)

# Import custom functions
from model import setup_model
from data import download_and_process_data

# Initialize model, tokenizer, and extractor
vision_text_model, tokenizer, extractor = setup_model()

# Load preprocessed dataset
dataset = download_and_process_data(tokenizer, extractor)

# Define training arguments
train_config = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir="./captioning-results",
)

metric_eval = evaluate.load("rouge")
ignore_pad_for_loss = True

def refine_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_eval_metrics(predictions):
    pred_texts, label_texts = predictions
    if isinstance(pred_texts, tuple):
        pred_texts = pred_texts[0]

    decoded_preds = tokenizer.batch_decode(pred_texts, skip_special_tokens=True)
    if ignore_pad_for_loss:
        label_texts = np.where(label_texts != -100, label_texts, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(label_texts, skip_special_tokens=True)

    # Post-processing
    decoded_preds, decoded_labels = refine_text(decoded_preds, decoded_labels)

    result = metric_eval.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    pred_lengths = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in pred_texts]
    result["generated_length"] = np.mean(pred_lengths)
    
    return result

# Trainer setup
trainer = Seq2SeqTrainer(
    model=vision_text_model,
    tokenizer=extractor,
    args=train_config,
    compute_metrics=compute_eval_metrics,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=default_data_collator,
)

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('punkt_tab')
    trainer.train()
    trainer.save_model("./captioning-results")
    tokenizer.save_pretrained("./captioning-results")

    # Perform inference
    caption_pipeline = pipeline("image-to-text", model="./captioning-results")
    print(caption_pipeline("test_image.jpg"))
