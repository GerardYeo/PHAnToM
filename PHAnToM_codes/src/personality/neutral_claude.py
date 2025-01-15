import boto3
import pandas as pd
import pickle
import os
import json
from tqdm import tqdm
import time


# File params
ITEMPATH = "/home/XXXX-1/psych_reasoning/data/mpi_120.csv"
OUTPATH = "/home/XXXX-1/psych_reasoning/outs/claude2:1_mpi_120_run0.pickle"
MODEL_NAME = "anthropic.claude-v2:1"
TEST_TYPE = None
LABEL_TYPE = None

# Model params
lag_per_iter = 1
temperature = 0
max_tokens_to_sample = 400
top_p = 0.95


# Initialize client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    endpoint_url='https://bedrock.us-east-1.amazonaws.com'
    )

def getItems(filename=ITEMPATH, item_type=None, label_type=LABEL_TYPE):
    data = pd.read_csv(filename)
    if label_type is not None:
        items = data[data["label_ocean"] == label_type]
    else:
        items = data
    return items


template = """Human:
Question:
Given a statement of you: "You {}."
Please choose from the following options to identify how accurately this statement describes you.
Options:
(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate
Please reply with ONLY 1 alphabet representing your choice.

Assistant:"""


dataset = getItems(ITEMPATH, TEST_TYPE)

result = []
for i,row in tqdm(dataset.iterrows(), total=len(dataset)):
    prompt = template.format(row["text"].lower())
    
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": max_tokens_to_sample,
        "temperature": temperature,
        "top_p": top_p 
    })
    response = bedrock.invoke_model(
        body=body, 
        modelId=MODEL_NAME, 
        accept='application/json', 
        contentType='application/json'
    )
    response_body = json.loads(response.get('body').read())
    result.append((i, prompt, response_body['completion']))
    # Add lag to not overload Bedrock API
    time.sleep(lag_per_iter)


with open(OUTPATH, "wb+") as f:
    pickle.dump(result, f)