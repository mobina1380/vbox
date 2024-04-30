from datasets import DatasetDict, Dataset
import os
vbox_folder_path = "vbox"
text_file_path = "transcriptshort.txt"

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    audio_texts = []
    for line in lines:
        parts = line.strip().split('"')
        if len(parts) >= 4:  
            audio_file = parts[1]
            text = parts[3].strip()
            audio_text = {"audio": audio_file, "sentence": text}
            audio_texts.append(audio_text)
    return audio_texts

def process_vbox_folder(vbox_folder_path, text_file_path):
    audio_texts = read_text_file(text_file_path)
    audio_paths = [os.path.join(vbox_folder_path, audio_text["audio"]) for audio_text in audio_texts]
    sentences = [audio_text["sentence"] for audio_text in audio_texts]
    return audio_paths, sentences

audio_paths, sentences = process_vbox_folder(vbox_folder_path, text_file_path)

split_ratio = int(len(audio_paths) * 0.9)  
train_dataset = Dataset.from_dict({"audio": audio_paths[:split_ratio], "sentence": sentences[:split_ratio]})
test_dataset = Dataset.from_dict({"audio": audio_paths[split_ratio:], "sentence": sentences[split_ratio:]})

common_voice = DatasetDict({"train": train_dataset, "test": test_dataset})

print(common_voice)
from transformers import WhisperConfig, AutoModelForSeq2SeqLM
from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="persian", task="transcribe")
input_str = common_voice["train"][0]['sentence']
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input:                 {input_str}")
print(f"Decoded w/ special:    {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal:             {input_str == decoded_str}")

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="persian", task="transcribe")
print(common_voice["train"][0])

from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
MAX_DURATION_IN_SECONDS = 30.0
MAX_INPUT_LENGTH = MAX_DURATION_IN_SECONDS * 16000
MAX_LABEL_LENGTH = 448
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    batch["input_length"] = len(batch["audio"])
    batch["labels_length"] = len(tokenizer(batch["sentence"], add_special_tokens=False).input_ids)
    return batch
def filter_inputs(input_length):
    """Filter inputs with zero input length or longer than 30s"""
    return 0 < input_length < MAX_INPUT_LENGTH
def filter_labels(labels_length):
    """Filter label sequences longer than max length (448)"""
    return labels_length < MAX_LABEL_LENGTH 

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)
common_voice = common_voice.filter(filter_inputs, input_columns=["input_length"])
common_voice = common_voice.filter(filter_labels, input_columns=["labels_length"])
common_voice = common_voice.remove_columns(['labels_length', 'input_length'])


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.generation_config.language = "persian"
model.generation_config.task = "transcribe"

model.generation_config.forced_decoder_ids = None
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tiny-persian",  
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=False,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
processor.save_pretrained(training_args.output_dir)

trainer.train()