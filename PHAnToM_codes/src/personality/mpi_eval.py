import os
import re
import json
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
from transformers import set_seed

if __package__ is None:
    import sys
    from os import path
    p = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.insert(0,p)

from src.reasoning.fantom_agents import FlanT5Agent, FlanUL2Agent, MistralAIAgent, ZephyrAgent, ClaudeAgent, LlamaAgent, FalconAgent, GPT3BaseAgent, ConversationalGPTBaseAgent
from src.personality.consts import p2_descriptions as their_p2_descriptions, naive_prompt
from src.personality.descriptions import p2_descriptions as our_p2_descriptions
# from src.reasoning.agents.gpt import GPT3BaseAgent, ConversationalGPTBaseAgent
# from src.reasoning.agents.together_ai import TogetherAIAgent


"""
/home/XXXX-3/anaconda3/envs/py310/bin/python \
src/personality/mpi_eval.py --model 

"""
MAX_EXAMPLES_TO_PARSE = None # set as None to use all, set >0 for prototyping
PROJECT_HOME = "/home/XXXX-3/PsychLLM" #Path(__file__).parent.resolve()
DATA_DIR = 'data'
DATA_DIR_PATH = os.path.join(PROJECT_HOME, DATA_DIR)
RANDOM_SEED = 99
random.seed(RANDOM_SEED)
set_seed(RANDOM_SEED)


class MPIDataset(Dataset):
    def __init__(self, texts, args):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        return text


class MPIEvalAgent():
    def __init__(self, args):
        self.args = args
        self.eval_dir_path = os.path.join(PROJECT_HOME, 'outs', 'personality')
        if args.p2_source is not None:
            self.eval_dir_path = os.path.join(self.eval_dir_path, args.p2_source)
            self.p2_descriptions = self.get_descriptions()
        if MAX_EXAMPLES_TO_PARSE is not None:
            self.eval_dir_path = os.path.join(self.eval_dir_path, 'sample')
        self.output_filename_suffix = '_{}_p-{}.json'.format(self.args.model, str(self.args.personality))
        self.load_mpi()
        self.setup_mpi()
        self.model = self.load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_descriptions(self):
        if self.args.p2_source=='p2_ours':
            return our_p2_descriptions
        elif self.args.p2_source=='p2_theirs':
            return their_p2_descriptions
        elif self.args.p2_source=='naive':
            return naive_prompt
        else:
            raise NotImplementedError(f'{self.args.p2_source} source is not defined.')
    
    def load_mpi(self):
        self.mpi_df = pd.read_csv(os.path.join(PROJECT_HOME, 'data', 'personality','mpi_120.csv'))

    def respond(self, prompt):
        response = self.model.interact(prompt)
        return response

    def load_model(self):
        if self.args.model=='gpt-3.5-turbo-instruct':
            model = GPT3BaseAgent({'engine': self.args.model, 'temperature': 0, 'top_p': 0.95, 'frequency_penalty': 0.0, 'presence_penalty': 0.0})
        elif self.args.model=='gpt-3.5-turbo-1106':
            model = ConversationalGPTBaseAgent({'model': self.args.model, 'temperature': 0, 'top_p': 0.95, 'frequency_penalty': 0.0, 'presence_penalty': 0.0})
        elif self.args.model.startswith('flan-t5'):
            model = FlanT5Agent(self.args)
        elif self.args.model.startswith('flan-ul2'):
            model = FlanUL2Agent(self.args)
        elif self.args.model.startswith("anthropic"):
            model = ClaudeAgent({'model_name': self.args.model, 'temperature': 0, 'top_p': 0.95, 'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'max_tokens_to_sample': 700})
        # elif self.args.model.endswith('-tg'):
        #     model = TogetherAIAgent(self.args.__dict__)
        elif self.args.model.startswith('mistral'):
            model = MistralAIAgent(self.args)
        elif self.args.model.startswith('zephyr'):
            model = ZephyrAgent(self.args)
        elif self.args.model.lower().startswith('llama'):
            model = LlamaAgent(self.args)
        elif self.args.model.startswith('falcon'):
            model = FalconAgent(self.args)
        else:
            raise NotImplementedError

        return model

    def evaluate_response(self, responses):
        print("Running evaluation...")

        count = {'A' : 0, 'B' : 0, 'C' : 0, 'D' : 0, 'E': 0,'UNK' : 0}
        traits = {
            "O" : [],
            "C" : [],
            "E" : [],
            "A" : [],
            "N" : []
        }

        SCORES = {
            "A" : 5 , 
            "B" : 4 ,
            "C" : 3 ,
            "D" : 2 ,
            "E" : 1 ,
        }

        assert len(self.mpi_df) == len(responses), "Number of questions and model predictions should be the same."

        for loc, res in enumerate(responses):
            res = res.split('Answer:')[-1].strip().split('\n')[0]
            # choice = re.search(r'\([abcdeABCDE]\)', res, flags = 0).group()[1].upper()
            choice = re.search(r'[abcdeABCDE][^a-zA-Z]{0,}', res, flags = 0)
            if choice is None:
                count["UNK"] += 1
            else:
                choice = choice.group()[0].upper()
                count[choice] += 1

                row = self.mpi_df.loc[loc]
                label = row['label_ocean']
                label_raw = row['label_raw']
                key = row['key']
                score = SCORES[choice]

                if key == 1:
                    traits[label].append(score)
                else:
                    traits[label].append(6 - score)

        return (traits, count)

    def run_reports(self, evaluated_response):
        """
        Create report after scoring and analyzing the results        
        """
        traits, count = evaluated_response

        mean  = {}
        std  = {}
        for key, item in traits.items():
            mean[key] = np.mean(np.array(item))
            std[key] = np.std(np.array(item))

        return f'''{count}\n\nmean:\n {sorted(mean.items(), key=lambda item:item[0])}\n std:\n {sorted(std.items(), key=lambda item:item[0])}'''

    def dump_report_outputs(self, reports):
        """
        Dump the reports and the evaluation outputs
        """
        report_filename = "REPORT" + self.output_filename_suffix
        with open(os.path.join(self.eval_dir_path, report_filename), 'w') as f:
            json.dump(reports, f, indent=4)
        print(">>>>> REPORT filename: {}".format(report_filename))

    def setup_mpi(self):
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
        inputs = []
        for idx, row in self.mpi_df.iterrows():
            inputs.append(template.format(row["text"].lower()))
        self.inputs = inputs

    def parse_response(self, response):
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        elif "<ANSWER>" in response:
            match = re.search(r'<ANSWER>(.+?)</ANSWER>', response, re.DOTALL)
            if match:
                response = match.group(1)
            else:
                response = re.sub('<ANSWER>','',response)
        elif "Choose an answer from above:" in response:
            response = response.split("Choose an answer from above:")[-1].strip()

        return response

    def get_last_savepoint(self):
        responses_filename = "model_responses" + self.output_filename_suffix + "l" # jsonl
        model_responses_filename_path = os.path.join(self.eval_dir_path, responses_filename)

        # check if model outputs file exists
        if os.path.exists(model_responses_filename_path):
            print("File {} exists. Reading responses from file...".format(model_responses_filename_path))
            df = pd.read_json(model_responses_filename_path, lines=True)
            if len(df) > 0:
                last_idx = df.iloc[-1]['index']
                model_responses = df['response'].tolist()
            else:
                last_idx = -1
                model_responses = []
        else:
            last_idx = -1
            model_responses = []
        
        return last_idx, model_responses, model_responses_filename_path

    def run_batch_inference(self):
        mpi_dataset = MPIDataset(self.inputs, self.args)
        loader = DataLoader(mpi_dataset, batch_size=self.args.batch_size)

        model_responses = []
        print("Generating responses...")
        last_idx, model_responses, response_filename_path = self.get_last_savepoint()
        if last_idx > 0:
            last_idx = last_idx // self.args.batch_size
        for batch_idx, batch in enumerate(tqdm(loader)):
            if batch_idx <= last_idx:
                continue

            if self.args.personality is not None:
                personality_prefix = "Imagine you are someone that fits this description: " + self.p2_descriptions[self.args.personality.title()] + "\n\n"
                batch = [personality_prefix + b for b in batch]

            responses = self.model.batch_interact(batch)

            for idx, response in enumerate(responses):
                response = self.parse_response(response)
                model_responses.append(response)

                # save the model responses in a file on the fly
                with open(response_filename_path, 'a') as f:
                    instance_for_dump = {'index': batch_idx * self.args.batch_size + idx, 'response': response, 'input_prompt': batch[idx]}
                    json.dump(instance_for_dump, f)
                    f.write("\n")

        return model_responses

    def run_inference(self):
        target_data = self.inputs
        model_responses = []

        # check if the file exists
        last_idx, model_responses, response_filename_path = self.get_last_savepoint()

        print("Generating responses...")
        for idx, input_prompt in enumerate(tqdm(target_data)):
            if idx <= last_idx:
                continue

            if self.args.personality is not None:
                personality_prefix = "Imagine you are someone that fits this description: " + self.p2_descriptions[self.args.personality.title()] + "\n\n"
                input_prompt = personality_prefix + input_prompt

            response = self.model.interact(input_prompt)
            response = self.parse_response(response)
            model_responses.append(response)

            # save the model responses in a file on the fly
            with open(response_filename_path, 'a') as f:
                json.dump({'index': idx, 'input_prompt': input_prompt, 'response': response}, f)
                f.write("\n")

        return model_responses

    def run(self):
        os.makedirs(self.eval_dir_path, exist_ok=True)
        if args.existing_response_file_name is None:
            if self.args.model.startswith("gpt-") or self.args.model.startswith("anthropic") or self.args.model.startswith("text-") or self.args.model.endswith("-tg"):
                model_responses = self.run_inference()
            else:
                model_responses = self.run_batch_inference()
        else:
            print(">>> Reading responses from file...")
            model_responses = self.get_responses_from_file(self.args.existing_response_file_name)

        evaluated_outputs = self.evaluate_response(model_responses)
        reports = self.run_reports(evaluated_outputs)
        self.dump_report_outputs(reports)

    def get_responses_from_file(self, response_filename):
        setup = response_filename.removeprefix("model_responses").removesuffix(".jsonl")
        assert setup == self.output_filename_suffix.removesuffix(".json"), "The response file name does not match the output file name"

        response_file = os.path.join(self.eval_dir_path, response_filename)
        df = pd.read_json(response_file, lines=True)
        model_responses = df['response'].to_list()
        return model_responses


def main(args):
    evaluator = MPIEvalAgent(args)
    evaluator.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for generating dialogues')
    parser.add_argument('--model',
                        type=str,
                        default='gpt-4-0314',
                        choices=[
                            'flan-ul2', 'flan-t5-xxl', 'flan-t5-xl', 
                            'Llama-2-7b-hf', 'Llama-2-7b-chat-hf', 'Llama-2-13b-hf', 'Llama-2-13b-chat-hf',
                            'zephyr-7b-alpha', 'zephyr-7b-beta', 
                            'mistral', 'mistral-instruct', 
                            'mpt-30b-instruct-tg', 'guanaco-33b-tg', 
                            'anthropic.claude-v2:1','falcon-7b-instruct',
                            'gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-1106'
                            ],
                        help='name of the model to run evaluation',
    )
    parser.add_argument('--batch-size',
                        type=int,
                        default=1,
                        help='batch size for evaluation',
    )
    parser.add_argument('--existing-response-file-name',
                        type=str,
                        help='name of the response file that you want to recompute the report for',
    )
    parser.add_argument('--personality',
                        type=str,
                        default=None,
                        help='whether to use personality or None',
    )
    parser.add_argument('--p2_source',
                        type=str,
                        default=None,
                        choices=['naive','p2_theirs', 'p2_ours'],
                        help='which personality description to use',
    )

    args = parser.parse_args()
    main(args)
