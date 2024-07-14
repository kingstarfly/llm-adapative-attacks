import openai
from openai import OpenAI
import os
import time
import torch
import gc
import tiktoken
from typing import Dict, List
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.inference._client import TextGenerationOutput

load_dotenv()

# import google.generativeai as palm

from openai import AzureOpenAI


class GPT:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_LOGPROBS = True
    API_TOP_LOGPROBS = 20

    def __init__(self, model_name):
        self.model_name = model_name
        if "4o" in model_name:
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2023-05-15",
                azure_endpoint="https://azure-openai-oscegen-east-us.openai.azure.com/",
            )
        elif "gpt" in model_name:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif "together" in model_name:
            TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
            self.client = OpenAI(
                api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz"
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")  # same as for gpt-3.5
        self.tokenizer.vocab_size = 100256  # note values from 100256 to 100275 (tokenizer.max_token_value) throw an error

    def generate(
        self,
        convs: List[List[Dict]],
        max_n_tokens: int,
        temperature: float,
        top_p: float,
    ):
        """
        Args:
            convs: List of conversations (each of them is a List[Dict]), OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        outputs = []
        for conv in convs:
            output = self.API_ERROR_OUTPUT
            for _ in range(self.API_MAX_RETRY):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=conv,
                        max_tokens=max_n_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=self.API_TIMEOUT,
                        logprobs=self.API_LOGPROBS,
                        top_logprobs=self.API_TOP_LOGPROBS,
                        seed=0,
                    )
                    response_logprobs = [
                        dict(
                            (
                                response.choices[0]
                                .logprobs.content[i_token]
                                .top_logprobs[i_top_logprob]
                                .token,
                                response.choices[0]
                                .logprobs.content[i_token]
                                .top_logprobs[i_top_logprob]
                                .logprob,
                            )
                            for i_top_logprob in range(self.API_TOP_LOGPROBS)
                        )
                        for i_token in range(len(response.choices[0].logprobs.content))
                    ]
                    output = {
                        "text": response.choices[0].message.content,
                        "logprobs": response_logprobs,
                        "n_input_tokens": response.usage.prompt_tokens,
                        "n_output_tokens": response.usage.completion_tokens,
                    }
                    break
                except openai.OpenAIError as e:
                    print(type(e), e)
                    time.sleep(self.API_RETRY_SLEEP)

                time.sleep(self.API_QUERY_SLEEP)
            outputs.append(output)
        return outputs


class HuggingFaceInference:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 60

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.client = InferenceClient(token=self.api_key)

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[Dict]:
        outputs = []
        for prompt in prompts:
            output = self.API_ERROR_OUTPUT
            for _ in range(self.API_MAX_RETRY):
                try:
                    response = self.client.text_generation(
                        prompt,
                        model=self.model_name,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        details=True,
                        return_full_text=False,
                    )

                    output = self._process_response(response, prompt)
                    break
                except Exception as e:
                    print(f"Error: {type(e).__name__} - {str(e)}")
                    time.sleep(self.API_RETRY_SLEEP)

                time.sleep(self.API_QUERY_SLEEP)
            outputs.append(output)
        return outputs

    def _process_response(self, response: TextGenerationOutput, prompt: str) -> Dict:
        return {
            "text": response.generated_text,
            "logprobs": self._extract_logprobs(response),
            "n_input_tokens": len(response.details.prefill),
            "n_output_tokens": len(response.details.tokens),
        }

    def _extract_logprobs(self, response: TextGenerationOutput) -> List[Dict]:
        logprobs = []
        for token_info in response.details.tokens:
            token_logprobs = {}
            for i, (token, score) in enumerate(
                zip(token_info.top_tokens, token_info.top_logprobs)
            ):
                if (
                    i >= 20
                ):  # Limit to top 20 logprobs to match the original implementation
                    break
                token_logprobs[token] = score
            logprobs.append(token_logprobs)
        return logprobs


# Usage example:
# hf_inference = HuggingfaceInference("gpt2")
# results = hf_inference.generate(["Hello, how are you?"], max_new_tokens=50, temperature=0.7, top_p=0.9)


class HuggingFace:
    def __init__(self, model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        # substitute '▁Sure' with 'Sure' (note: can lead to collisions for some target_tokens)
        self.pos_to_token_dict = {
            v: k.replace("▁", " ") for k, v in self.tokenizer.get_vocab().items()
        }
        # self.pos_to_token_dict = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        if "llama3" in self.model_name.lower():
            self.eos_token_ids.append(
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            )
        self.pad_token_id = self.tokenizer.pad_token_id

    def generate(
        self,
        full_prompts_list: List[str],
        max_n_tokens: int,
        temperature: float,
        top_p: float = 1.0,
    ) -> List[Dict]:

        if "llama2" in self.model_name.lower():
            max_n_tokens += 1  # +1 to account for the first special token (id=29871) for llama2 models
        batch_size = len(full_prompts_list)
        vocab_size = len(self.tokenizer.get_vocab())
        inputs = self.tokenizer(full_prompts_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(torch.device("mps").index) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        # Batch generation
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_n_tokens,
            do_sample=False if temperature == 0 else True,
            temperature=None if temperature == 0 else temperature,
            eos_token_id=self.eos_token_ids,
            pad_token_id=self.tokenizer.pad_token_id,  # added for Mistral
            top_p=top_p,
            output_scores=True,
            return_dict_in_generate=True,
        )
        output_ids = output.sequences
        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, input_ids.shape[1] :]
        if "llama2" in self.model_name.lower():
            output_ids = output_ids[:, 1:]  # ignore the first special token (id=29871)

        generated_texts = self.tokenizer.batch_decode(output_ids)
        # output.scores: n_output_tokens x batch_size x vocab_size (can be counter-intuitive that batch_size doesn't go first)
        logprobs_tokens = [
            torch.nn.functional.log_softmax(output.scores[i_out_token], dim=-1)
            .cpu()
            .numpy()
            for i_out_token in range(len(output.scores))
        ]
        if "llama2" in self.model_name.lower():
            logprobs_tokens = logprobs_tokens[
                1:
            ]  # ignore the first special token (id=29871)

        logprob_dicts = [
            [
                {
                    self.pos_to_token_dict[i_vocab]: logprobs_tokens[i_out_token][
                        i_batch
                    ][i_vocab]
                    for i_vocab in range(vocab_size)
                }
                for i_out_token in range(len(logprobs_tokens))
            ]
            for i_batch in range(batch_size)
        ]

        outputs = [
            {
                "text": generated_texts[i_batch],
                "logprobs": logprob_dicts[i_batch],
                "n_input_tokens": len(
                    input_ids[i_batch][input_ids[i_batch] != 0]
                ),  # don't count zero-token padding
                "n_output_tokens": len(output_ids[i_batch]),
            }
            for i_batch in range(batch_size)
        ]

        for key in inputs:
            inputs[key].to("cpu")
        output_ids.to("cpu")
        del inputs, output_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return outputs
