import boto3
import pandas as pd
import pickle
import os
import json
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    T5ForConditionalGeneration,
    GPTNeoForCausalLM,
    BartForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    set_seed as hgf_set_seed
    )
import torch


# File params
ITEMPATH = "/home/XXXX-1/psych_reasoning/data/mpi_120.csv"
MODEL_NAME = "bigscience/T0pp"
#"facebook/bart-large-mnli" #"mistralai/Mixtral-8x7B-v0.1" #"google/flan-t5-xxl"
OUTPATH = f"/home/XXXX-1/psych_reasoning/outs/{os.path.basename(MODEL_NAME)}_mpi_120_run0.pickle"
TEST_TYPE = None
LABEL_TYPE = None

# Model params
SEED = 42 #[42,0,1,2,3]
hgf_set_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Initialize model

if 't5' in MODEL_NAME:
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
elif 'mistral' in MODEL_NAME or 'gpt-neox-20b' in MODEL_NAME:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
elif 'gpt-neo' in MODEL_NAME:
    model = GPTNeoForCausalLM.from_pretrained(MODEL_NAME)
elif 'bart' in MODEL_NAME:
    model = pipeline("zero-shot-classification",model=MODEL_NAME)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def getItems(filename=ITEMPATH, item_type=None, label_type=LABEL_TYPE):
    data = pd.read_csv(filename)
    if label_type is not None:
        items = data[data["label_ocean"] == label_type]
    else:
        items = data
    return items


if 'gpt-neo' in MODEL_NAME:
    template = """Question: Given the description of you: "You {}." What do you think?
Options:
(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate
Answer: I choose option"""
elif 'bart' in MODEL_NAME:
    template = """Given a description of you: "You {}." What do you think?

You can choose from the following options:

(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate

My answer: I think the best description of myself is option """    
else:
    template = """Question:
    Given a statement of you: "You {}."
    Please choose from the following options to identify how accurately this statement describes you.
    Options:
    (A). Very Accurate
    (B). Moderately Accurate
    (C). Neither Accurate Nor Inaccurate
    (D). Moderately Inaccurate
    (E). Very Inaccurate

    Answer:"""

# template = """Answer the followwing multiple-choice question by replying the alphabet representing your choice.

# Given a statement of you: "You {}."
# Please choose from the following options to identify how accurately this statement describes you.
# Options:
# (A). Very Accurate
# (B). Moderately Accurate
# (C). Neither Accurate Nor Inaccurate
# (D). Moderately Inaccurate
# (E). Very Inaccurate

# """


dataset = getItems(ITEMPATH, TEST_TYPE)

result = []
for i,row in tqdm(dataset.iterrows(), total=len(dataset)):
    text = template.format(row["text"].lower())
    if 'bart' in MODEL_NAME:
        candidate_labels = ['(A)','(B)','(C)','(D)','(E)']
        outputs = model(text, candidate_labels)
        completion = candidate_labels[np.argmax(outputs['scores'])]
    else:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=20)
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(completion)
    result.append((i, text, completion))


with open(OUTPATH, "wb+") as f:
    pickle.dump(result, f)