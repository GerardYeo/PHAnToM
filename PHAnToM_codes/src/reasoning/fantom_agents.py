import torch
import re
import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    T5Tokenizer, T5ForConditionalGeneration, 
    LlamaTokenizer, LlamaForCausalLM
    )

class HuggingFaceAgent():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess_text(self, text):
        return text

    def postprocess_output(self, response):
        return response

    def interact(self, text):
        prompt = self.preprocess_text(text)
        encoded_texts = self.tokenizer(prompt, truncation=True, return_tensors="pt", max_length=512)
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256)
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = self.postprocess_output(decoded_output)

        return response

    def batch_interact(self, batch_texts):
        batch_prompts = [self.preprocess_text(text) for text in batch_texts]
        encoded_texts = self.tokenizer(batch_prompts, padding='longest', truncation=True, return_tensors="pt")
        input_ids = encoded_texts['input_ids'].to(self.device)
        attention_mask = encoded_texts['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [self.postprocess_output(decoded_output) for decoded_output in decoded_outputs]

        return responses

class FlanT5Agent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = T5Tokenizer.from_pretrained("google/" + args.model)
        self.model = T5ForConditionalGeneration.from_pretrained("google/" + args.model, device_map="auto")

class FlanUL2Agent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)

class MistralAIAgent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        if 'instruct' in self.args.model.lower():
            model_name = "Mistral-7B-Instruct-v0.1"
        else:
            model_name = "Mistral-7B-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/" + model_name)
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/" + model_name, device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token 

    def preprocess_text(self, text):
        text = re.sub("\nAnswer:","",text)
        return f"[INST]{text}[/INST]\n\nAnswer:" #self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)

    def postprocess_output(self, response):
        response = response.split("[/INST]")[-1].strip()
        if '[INST]' in response: # If Mistral starts "new instructions"
            response = response.split("[INST]")[0]
        return response

class ZephyrAgent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/" + self.args.model)
        self.model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/" + self.args.model, device_map="auto")

    def preprocess_text(self, text):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)

    def postprocess_output(self, response):
        return response.split("\n<|assistant|>\n")[-1].strip()

class LlamaAgent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args) 
        # need to request for access first with your account
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/" + self.args.model)
        self.model = LlamaForCausalLM.from_pretrained("meta-llama/" + self.args.model, device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess_text(self, text):
        text = re.sub("\nAnswer:","",text)
        return f"[INST]{text}[/INST]\n\nAnswer:" #self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)

    def postprocess_output(self, response):
        response = response.split("[/INST]")[-1].strip()
        if '[INST]' in response: # If Mistral starts "new instructions"
            response = response.split("[INST]")[0]
        return response


class FalconAgent(HuggingFaceAgent):
    def __init__(self, args):
        super().__init__(args)
        model_path = "tiiuae/" + self.args.model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        
    def interact(self, text:str):
        prompt = self.preprocess_text(text)
        with torch.no_grad():
            output = self.pipeline(prompt, 
                                   max_new_tokens=256,
                                   eos_token_id=self.tokenizer.eos_token_id,
                                   return_full_text=False,
                                  )
        output = output[0]['generated_text']
        response = self.postprocess_output(output)
        return response
    
    def batch_interact(self, batch_texts):
        batch_prompts = [self.preprocess_text(text) for text in batch_texts]
        with torch.no_grad():
            output = self.pipeline(batch_prompts, 
                                   max_new_tokens=256,
                                   eos_token_id=self.tokenizer.eos_token_id,
                                   return_full_text=False,
                                  )
        output = [x[0]['generated_text'] for x in output]
        responses = [self.postprocess_output(x) for x in output]
        return responses
    
    def postprocess_output(self, response:str) -> str:
        return response.strip()
    
    def preprocess_text(self, text:str) -> str:
        text = text.replace('Question:', '>>QUESTION<<')
        text = text.replace('Answer:', '>>ANSWER<<')
        return text

####################

import os
import time
import openai
from types import SimpleNamespace

class GPT3BaseAgent():
    def __init__(self, kwargs: dict):
        openai.api_key = ''
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()

    def _set_default_args(self):
        if not hasattr(self.args, 'engine'):
            self.args.engine = None # force user to set this, will throw error if not set
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0.9
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 0.9
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0.7
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0

    def generate(self, prompt):
        while True:
            try:
                completion = openai.Completion.create(
                    engine=self.args.engine,
                    prompt=prompt,
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens,
                    top_p=self.args.top_p,
                    frequency_penalty=self.args.frequency_penalty,
                    presence_penalty=self.args.presence_penalty,
                    stop=self.args.stop_tokens if hasattr(self.args, 'stop_tokens') else None,
                    logprobs=self.args.logprobs if hasattr(self.args, 'logprobs') else 0,
                    echo=self.args.echo if hasattr(self.args, 'echo') else False
                )
                break
            except (RuntimeError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.APIConnectionError) as e:
                print("Error: {}".format(e))
                time.sleep(2)
                continue

        return completion

    def parse_basic_text(self, response):
        output = response['choices'][0]['text'].strip()

        return output

    def parse_ordered_list(self, numbered_items):
        ordered_list = numbered_items.split("\n")
        output = [item.split(".")[-1].strip() for item in ordered_list if item.strip() != ""]

        return output

    def interact(self, prompt):
        response = self.generate(prompt)
        output = self.parse_basic_text(response)

        return output


class ConversationalGPTBaseAgent(GPT3BaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

    def _set_default_args(self):
        if not hasattr(self.args, 'model'):
            self.args.model = "gpt-3.5-turbo-1106"
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0.9
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 0.9
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0.7
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0

    def generate(self, prompt):
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=self.args.model,
                    messages=[{"role": "user", "content": "{}".format(prompt)}]
                )
                break
            except (openai.error.APIError, openai.error.RateLimitError) as e: 
                print("Error: {}".format(e))
                time.sleep(2)
                continue

        return completion

    def parse_basic_text(self, response):
        output = response['choices'][0].message.content.strip()

        return output

####################

import json
import boto3
import botocore
from botocore.client import Config as BotoConfig
from types import SimpleNamespace

class ClaudeAgent():
    def __init__(self, kwargs: dict):
        self.args = SimpleNamespace(**kwargs)
        # Initialize client
        config = BotoConfig(
            read_timeout=600,
            connect_timeout=600,
            retries={"max_attempts": 0}
        )
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',
            endpoint_url='https://bedrock.us-east-1.amazonaws.com',
            config=config
            )


    def generate(self, prompt):
        # while True:
            # try:
        body = json.dumps({
            "prompt": f"\n\nHuman:{prompt}\n\nAssistant:",
            "max_tokens_to_sample": self.args.max_tokens_to_sample,
            "temperature": self.args.temperature
            })
        response = self.bedrock.invoke_model(
            body=body, 
            modelId=self.args.model_name, 
            accept='application/json', 
            contentType='application/json'
            )
        response = json.loads(response.get('body').read())
            #     # response_body is a dict with keys: 'completion', 'stop_reason', 'stop'  
            #     break
            # except (RuntimeError, botocore.exceptions.ReadTimeoutError, botocore.exceptions.ClientError, botocore.errorfactory.ValidationException) as e:
            #     print("Error: {}".format(e))
            #     time.sleep(2)
            #     continue

        return response

    def parse_basic_text(self, response):
        output = response['completion']
        return output

    # def parse_ordered_list(self, numbered_items):
    #     ordered_list = numbered_items.split("\n")
    #     output = [item.split(".")[-1].strip() for item in ordered_list if item.strip() != ""]
    #     return output

    def interact(self, prompt):
        response = self.generate(prompt)
        output = self.parse_basic_text(response)
        return output
