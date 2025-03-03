#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
'''
@File    :   InfiniteRetrieval.py
@Time    :   2024/11/18 18:01:22 
@Author  :   MrYXJ
@Version :   0.1
@Contact :   yxj2017@gmail.com
@Desc    :   
'''
import os
import time
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import DynamicCache

from .chunks import Chunks
from .retrieval import Retrieval


class Pipline:
    def __init__(self, 
                 config=None, 
                 model=None, tokenizer=None) -> None:

        self.default_config = {
                "llm_name_or_path": "D:\Code\models\Qwen2.5-0.5B-Instruct",
                "data_format_map": {"context": "context","question": "question"},  
                "retri_split_pattern": "(\n\n)|([。！？.?!])",
                "topk": 200,
                "chunk_size": 1024,
                "use_question_weight": True,
                "record_attention_path": None,  # malpot attention weights cost much time
                "record_retrieval_path": None,  
                "record_result_path": None,
                "compute_score_method": 3,
                "phrase_token_num": 8,
                "generate_method": "token_id",
                "add_prompt": True,
                "use_past_key_value": False  # no finetuing suggest no ues
        }

        if config is not None:
            self.config = config
        else:
            file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ir_config.json")
            if os.path.exists(file_path):
                self.config = json.load(open(file_path, "r"))
                # print("Loading InfiniRetireval's config from %s" % file_path)
                
        # complete config params
        for key, value in self.default_config.items():
            self.config[key] = self.config.get(key, self.default_config[key])
                
        self.data_format_map = self.config.get("data_format_map")
        # The pattern of text segmentation(re.split) during retrival, which detemine the granularity of retrival unit.
        # self.retri_split_pattern = r"([。！？；.?!;])" # origin pattern from LLMxReduce
        # self.retri_split_pattern = r"([。！？.?!])|(?<![\n。！？.?!])(\n\n)(?![\n。！？.?!])" # add '\n\n' and delete ';'
        # self.retri_split_pattern = r"([。！？.?!])|(?<![\n。！？.?!])(\n)(?![\n。！？.?!])" # add '\n\n' and delete ';
        # self.retri_split_pattern = "([。！？.?!\n])"
        # self.retri_split_pattern = "([：。！？:.?!\n])"
   
        self.retri_split_pattern = self.config.get("retri_split_pattern")  # The pattern of text segmentation(re.split) during retrival, which detemine the granularity of retrival unit.

        self.topk = self.config["topk"]
        self.phrase_token_num = self.config["phrase_token_num"]
        self.chunk_size = self.config["chunk_size"]
        
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.get("llm_name_or_path"))
        else:
            self.tokenizer = tokenizer
        if model is None:
            if "Qwen" in self.config.get("llm_name_or_path"):
                self.config["torch_dtype"] = torch.bfloat16
            else:
                self.config["torch_dtype"] = torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.get("llm_name_or_path"), 
                    device_map="auto", 
                    attn_implementation="eager",
                    torch_dtype=self.config["torch_dtype"],
                )
        else:
            self.model = model
        
        self.config["torch_dtype"] = self.model.dtype
        assert self.model.config._attn_implementation == "eager", """Please set attn_implementation="eager" in AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager"), because Our Infini-Retri currently only supports tranditional attention-based model to inference."""

        # print("model attn_implementation:", self.model.config._attn_implementation)
        self.model.eval()                                            
        self.chunks = Chunks(self.config, self.tokenizer)
        self.retri = Retrieval(self.config, self.model, self.tokenizer)
        self.cache = None
        
        self.use_question_weight = self.config.get("use_question_weight")
        self.record_attention = self.config.get("record_attention_path")
        self.record_retrieval = self.config.get("record_retrieval_path")
        self.record_result = self.config.get("record_result_path")
        # print("Ininitial InfiniRetrieval's Config:", self.config)

        
    def UpdateCache(self, select_index, Cache):
        layer_num = len(Cache)
        Shape = Cache[0][0].shape
        Device = Cache[0][0].device
        num_key_value_heads, head_dim = Shape[1], Shape[3]
        select_index = torch.tensor(select_index).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        select_index = select_index.expand(-1, num_key_value_heads, -1, head_dim).to(Device)
        new_cache = []
        for layer in range(layer_num):
            key_tmp, value_tmp = Cache[layer][0], Cache[layer][1]
            select_key = key_tmp.gather(dim=2, index=select_index)
            select_value = value_tmp.gather(dim=2, index=select_index)
            new_cache.append((select_key, select_value))

        return tuple(new_cache)


    def record_info(self, record_path, record_file, print_info):
        if not os.path.exists(record_path):
            os.makedirs(record_path)
        
        file_path = os.path.join(record_path, record_file)
        with open(file_path, mode="a+", encoding="utf-8") as f:
            f.write(
                json.dumps(print_info, ensure_ascii=False)+'\n'
            )

    
    def generate_init(self):
        self.cache = None
        self.retri.PREFIX_PROMPT_TOKEN_NUM = 0
        self.retri.LAST_PROMPT_TEXT = None
        self.retri.START_TIME = time.time()
        self.retri.thoughts.malplot_all_init()
        self.retri.QUESTION_WEIGHT = None
    

    # def generate(self, data_dict, prompt, gen_name, 
    #              sample_id=0, topk=None,
    #              retri_split_pattern=None,
    #              temperature=0.1, max_new_tokens=128):
                 
    #     self.generate_init()

    #     t1 = time.time()
        
    #     # question = data_dict[self.data_format_map["question"]]
        
    #     docs = self.chunks.chunk_docs(data_dict[self.data_format_map["context"]], 
    #         chunk_size=self.chunk_size, question=None)

    #     chunk_cost_time = time.time() - t1
    #     tqdm_desc = "chunked cost=%.1fs size=%d" % (chunk_cost_time, self.chunk_size)
        
    #     origin_token_length = len(self.tokenizer.encode(data_dict[self.data_format_map["context"]], add_special_tokens=False))
    #     past_split_tokens, past_token_num = [], 0

    #     # for index, context in enumerate(docs):
    #     for index, context in enumerate(tqdm(docs, desc=tqdm_desc)):
    #         doc_id = "%s_%d_%d" % (gen_name, sample_id, index)
    #         data_dict[self.data_format_map["context"]] = context
    #         output, select_indexs, past_split_tokens, process_info = self.retri.focus_key_sentence(
    #             data_dict, prompt, cache=self.cache, past_split_tokens=past_split_tokens,
    #             past_token_num=past_token_num, doc_id=doc_id, topk=self.topk if topk is None else topk,
    #             retri_split_pattern=self.retri_split_pattern if retri_split_pattern is None else retri_split_pattern)
            
    #         past_cache = output["past_key_values"]
    #         past_token_num = process_info["select_num"]
    #         self.cache = self.UpdateCache(select_indexs, past_cache)
            
    #         del output
    #         del past_cache
    #         torch.cuda.empty_cache()

    #         if self.record_retrieval:
    #             save_path = os.path.join(self.record_retrieval, "%s_%s" %(gen_name, sample_id))
    #             self.record_info(save_path, "process.json", process_info)

    #     retrieval_tokens = [token for sublist in past_split_tokens for token in sublist]
    #     retrieval_text = self.tokenizer.decode(retrieval_tokens)

    #     input_text = retrieval_text + self.retri.LAST_PROMPT_TEXT
    #     model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        
    #     t1 = time.time()
    #     with torch.no_grad(): 
    #         if self.config.get("generate_method", "token_id") == "past_key_value":
    #             generated_ids = self.model.generate(**model_inputs, past_key_values=self.cache, 
    #                 temperature=temperature, max_new_tokens=max_new_tokens)
    #         elif self.config.get("generate_method", "token_id") == "token_id":
    #             generated_ids = self.model.generate(**model_inputs, 
    #                 temperature=temperature, max_new_tokens=max_new_tokens)
    #     cost_time = time.time() - t1
        
    #     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)]
    #     response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    #     result_info = {"input_text": input_text, "output": response, 
    #                    "generate_method": self.config["generate_method"], 
    #                    "generate_cost": cost_time,
    #                    "model_name_or_path": self.model.name_or_path,
    #                    "input_token_length": model_inputs["input_ids"].shape[1],
    #                    "origin_token_length": origin_token_length,
    #                    "chunk_size": self.chunk_size, "retrieval_topk": self.topk,
    #                    "compute_score_method": str(self.retri.compute_score_method), 
    #                    "phrase_token_num" : str(self.retri.phrase_token_num)}
        
    #     # 有些模型输出中带有完整思考过程，并以<think></think>包裹，例如deepseek-qwen
    #     if "</think>" in result_info["output"]:
    #         output = result_info["output"]
    #         think_len = len("</think>")
    #         result_info["output"] = output[output.rfind("</think>")+think_len:]
    #         result_info["think"] = output[:output.rfind("</think>")+think_len]
    #         response = result_info["output"]
            
    #     if self.record_attention:
    #         save_name = "%s_%s/" % (gen_name, sample_id)
    #         self.retri.thoughts.malplot_all_attention(save_name=save_name, page_max_num=8)
            
    #     if self.record_result:
    #         save_path = os.path.join(self.record_result, "%s_%s" %(gen_name, sample_id))
    #         self.record_info(save_path, "result.json", result_info)

    #     return response, result_info
    

    def generate2(self, data_dict, prompt, gen_name, 
                  sample_id=0, topk=None, phrase_token_num=None,
                  retri_split_pattern=None, chunk_size=None,
                  temperature=0.1, max_new_tokens=128):
                 
        self.generate_init()

        t1 = time.time()
        
        # question = data_dict[self.data_format_map["question"]]
        
        docs = self.chunks.chunk_docs(data_dict[self.data_format_map["context"]], 
            chunk_size=self.chunk_size if chunk_size is None else chunk_size, question=None)

        chunk_cost_time = time.time() - t1
        # tqdm_desc = "chunked cost=%.1fs size=%d" % (chunk_cost_time, self.chunk_size)
        
        origin_token_length = len(self.tokenizer.encode(data_dict[self.data_format_map["context"]], add_special_tokens=False))
        past_split_tokens, past_token_num = [], 0
        past_retri_context = ""
        
        # for index, context in enumerate(docs):
        for index, context in enumerate(tqdm(docs, desc="Inferencing", colour="BLUE", leave=False)):
            doc_id = "%s_%d_%d" % (gen_name, sample_id, index)
            data_dict[self.data_format_map["context"]] = past_retri_context + context

            output, select_indexs, past_split_tokens, process_info = self.retri.focus_key_sentence(
                data_dict, prompt, cache=self.cache, past_split_tokens=past_split_tokens,
                past_token_num=past_token_num, doc_id=doc_id, topk=self.topk if topk is None else topk,
                phrase_token_num=self.phrase_token_num if phrase_token_num is None else phrase_token_num,
                retri_split_pattern=self.retri_split_pattern if retri_split_pattern is None else retri_split_pattern)

            past_token_num = process_info["select_num"]

            if self.config["use_past_key_value"]:
                past_cache = output["past_key_values"]
                self.cache = self.UpdateCache(select_indexs, past_cache)
                del output
                del past_cache
                torch.cuda.empty_cache()
            else:
                # 如果不使用past_key_value 在外面就先合并之前 retrieval context后与当前context一并作为input, 之后相当于第一次输入，past_token_num=0
                if past_token_num > 0:
                    past_retri_context = ""
                    for sen_tokens in past_split_tokens:
                        past_retri_context += self.tokenizer.decode(sen_tokens)
                past_token_num = 0
                past_split_tokens = []
        
            if self.record_retrieval:
                save_path = os.path.join(self.record_retrieval, "%s_%s" %(gen_name, sample_id))
                self.record_info(save_path, "process.json", process_info)
        
        if past_token_num !=0:
            retrieval_tokens = [token for sublist in past_split_tokens for token in sublist]
            retrieval_text = self.tokenizer.decode(retrieval_tokens)
            input_text = retrieval_text + self.retri.LAST_PROMPT_TEXT
        else:
            data_dict["context"] = past_retri_context
            input_text = self.retri.build_prompt(data_dict=data_dict, prompt=prompt)
            input_text = self.retri.build_message(input_text)
      
        model_inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        
        t1 = time.time()
        with torch.no_grad(): 
            if self.config.get("generate_method", "token_id") == "past_key_value":
                generated_ids = self.model.generate(**model_inputs, past_key_values=self.cache, 
                    temperature=temperature, max_new_tokens=max_new_tokens)
            elif self.config.get("generate_method", "token_id") == "token_id":
                generated_ids = self.model.generate(**model_inputs, 
                    temperature=temperature, max_new_tokens=max_new_tokens)
        cost_time = time.time() - t1
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        result_info = {"llm_name": self.config.get("llm_name_or_path"),
                       "llm_type": str(self.config["torch_dtype"]),
                       "input_text": input_text, "output": response, 
                       "generate_method": self.config["generate_method"], 
                       "generate_cost": cost_time,
                       "model_name_or_path": self.model.name_or_path,
                       "input_token_length": model_inputs["input_ids"].shape[1],
                       "origin_token_length": origin_token_length,
                       "chunk_size": self.chunk_size, "retrieval_topk": self.topk,
                       "compute_score_method": str(self.retri.compute_score_method), 
                       "phrase_token_num" : str(self.retri.phrase_token_num)}
        
        # 有些模型输出中带有完整思考过程，并以<think></think>包裹，例如deepseek-qwen
        if "</think>" in result_info["output"]:
            output = result_info["output"]
            think_len = len("</think>")
            result_info["output"] = output[output.rfind("</think>")+think_len:]
            result_info["think"] = output[:output.rfind("</think>")+think_len]
            response = result_info["output"]
            
        if self.record_attention:
            save_name = "%s_%s/" % (gen_name, sample_id)
            self.retri.thoughts.malplot_all_attention(save_name=save_name, page_max_num=8)
            
        if self.record_result:
            save_path = os.path.join(self.record_result, "%s_%s" %(gen_name, sample_id))
            self.record_info(save_path, "result.json", result_info)

        return response, result_info

