#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import re
import time

import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import DynamicCache

from .thoughts import thoughts

class Retrieval:
    def __init__(self,
                 config,
                 model=None, 
                 tokenizer=None):
        
        Path = config["llm_name_or_path"]
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(Path)
        else:
            self.tokenizer = tokenizer
        if model is None:
            self.model = AutoModelForCausalLM.from_pretrained(Path, device_map="auto",
                                                              attn_implementation="eager")
        else:
            self.model = model
        self.config = config
        # self.retri_split_pattern = self.config.get("retri_split_pattern", r"([。！？；.?!;])")
        self.retri_split_pattern = self.config.get("retri_split_pattern", "([：。！？:.?!\n])")
        self.record_attention = self.config.get("record_attention_path", None)
        
        # 第一个doc segment 输入时，记录整个input text部分中在context部分之前的template prompt的token数目，该数目和token在之后每次推理过程中是固定，即可直接使用past_kv_cache中token值，同时该数目还可以验证之后每次操作的准确性。在求得该值时需要保证能够正确切开 template prompt与context部分。
        self.PREFIX_PROMPT_TOKEN_NUM = 0
        self.START_TIME = time.time()
        self.LAST_PROMPT_TEXT = None
        self.use_question_weight = self.config.get("use_question_weight", False) # 使用有待在更多数据，更多模型上进行验证
        self.QUESTION_WEIGHT = None
        # retrieval token and analyse process function
        self.compute_score_method = self.config.get("compute_score_method", 2)
        self.phrase_token_num = self.config.get("phrase_token_num", 0)

        self.thoughts = thoughts(config={"plt_save": self.record_attention})
      

    # def split_sentence(self, text, spliter=r'([。！？；.?!;])'):
    # def split_sentence(self, text, spliter):
    #     """
    #     Split by punctuation and keep punctuation
    #     """
    #     sentence_list = re.split(spliter, text)
    #     sentence_list = list(filter(None, sentence_list)) # filter 

    #     # Rearrange sentences and punctuation
    #     if spliter != ' ':
    #         sentences = ["".join(i) for i in zip(
    #             sentence_list[0::2], sentence_list[1::2])]
    #         if len(sentence_list) % 2 != 0 and sentence_list[-1] != '':
    #             sentences.append(sentence_list[-1])
    #     else:
    #         sentences = [i+' ' for i in sentence_list if i != '']
    #         sentences[-1] = sentences[-1].strip()
    #     return sentences
    

    def split_sentence(self, text, spliter):
        """
        Split by punctuation(in spliter) and keep punctuation
        """
        sentence_list = re.split(spliter, text)
        sentence_list = list(filter(None, sentence_list)) # filter 
        punctuation = ["\n\n", "。", "！", "？", ".", "?", "!"]
        # Rearrange sentences and punctuation
        if spliter != ' ':
            sentences = []
            cur_sen = ""
            for sen in sentence_list:
                # 下面写法保证根据符号切分之后，继续拼接符号并尽量保持之前格式。   
                # 处理当英文字母中名字姓氏之间的'.'不切分（少于9个字母不切）
                if len(cur_sen.strip()) == 0 \
                or len(sen.strip()) == 0 or sen.strip() in punctuation \
                or cur_sen[-1] == '.' and len(sen.strip()) <= 8:
                    cur_sen += sen
                else:
                    sentences.append(cur_sen)
                    cur_sen = sen
            if len(cur_sen.strip()) > 0:
                sentences.append(cur_sen)
        else:
            sentences = [i+' ' for i in sentence_list if i != '']
            sentences[-1] = sentences[-1].strip()
        return sentences

    
    def verify_text_index(self, verify_text, input_ids, 
                          start_index, end_index, spliter):
        """To check the index position is the correct location for the text

        Args:
            verity_text (str): question or context in origin input text
            input_ids (token ids): question or context in input token ids
            start_index (int): question or context segement start index
            end_index (int): question or context segment end index

        Returns:
            bool: whether the index position is the correct location for the text  
        """
        verify_ids = []
        for text in self.split_sentence(verify_text, spliter):
            verify_ids.extend(self.tokenizer.encode(text, add_special_tokens=False))
        decode_verify_text = self.tokenizer.decode(verify_ids)

        input_text = self.tokenizer.decode(input_ids[start_index:end_index])
        
        #纯粹是判断数据中出现大量.与puncatution中符号结合的脏数据情况
        if input_text.startswith("<text>\n\n"):
            input_text = input_text.replace("<text>\n\n", "")
            if input_text == decode_verify_text[:len(input_text)]:
                return True
        
        if input_text.strip() == decode_verify_text.strip():
            return True
        else:
            print("input_text:", input_text.strip())
            print("context:", decode_verify_text.strip())
            print("Input ids Segment:[%d, %d]" % (start_index, end_index))
            # print("Input ids text.strip():", input_text.strip())
            # print("Re decode text.strip():", decode_verify_text.strip())
            return False


    def build_prompt(self, *args, **kwargs):
        """You need wrapper a function to make prompt template correct.
        """
        data = kwargs["data_dict"]
        for key in data.keys():
            data[key] = data[key].strip()

        template = kwargs["prompt"]
        prompt = template.format(**data)
        return prompt


    # def build_default_prompt(self, ):
    #     data = kwargs["data_dict"]
    #     template = kwargs["prompt"]
    #     prompt = template.format(**data)
    #     return prompt
    
    
    def build_message(self, text): # qwen2 prompt format
        message = [
            {"role": "system", "content": ""},
            {"role": "user", "content": text}
        ]
        # message = [{'role': 'user', 'content': text}]
        message_str = self.tokenizer.apply_chat_template(
            conversation=message, tokenize=False, add_generation_prompt=True)
        return message_str
    
    
    def decode_sentence(self, split_tokens):
        sentences = []
        for tokens in split_tokens:
            sentences.append(self.tokenizer.decode(tokens))
        return sentences

    
    def merge_past_now(self, past_split_tokens, now_split_texts):
        """ merge past select text and now input text into completed input text
        """
        merge_split_texts = self.decode_sentence(past_split_tokens)

        total = 0
        for sentence in now_split_texts:
            total += len(self.tokenizer.encode(sentence, add_special_tokens=False))
            if total > self.PREFIX_PROMPT_TOKEN_NUM:
                merge_split_texts.append(sentence)
        return merge_split_texts
    

    def pre_process(self, data_dict, prompt, 
                    spliter=None, past_split_tokens=[], past_token_num=0):
        """
        In Order to obtain the segmentation position of each sentence after the text is converted into a token sequence, 
            the sentences are segmented according to the sentence punctuation marks, and the tokenizer is encoded into a token senquence and conbined.
        In addition, this step also needs to deal with findding the question and context's segment(start, end) index of the entire token, 
            which is very important for retrievaling the correct result in the Attention Scores.
        Args:
            # spliter: Two \n\n in a row when there is no punctuation('[。！？.?!]') are also used as sentence separators.
            spliter: default: "(\n{2,})|([：。！？:.?!])"
            spliter: default: 
        """

        spliter = self.retri_split_pattern # if spliter is None else spliter        
        input_text = self.build_prompt(data_dict=data_dict, prompt=prompt)  # 用户根据不同任务prompt template 先自行定义
        assert input_text is not None, "You need wrapper a function to make prompt template correct."
        
        input_text = self.build_message(input_text)
        split_texts = self.split_sentence(input_text, spliter=spliter)
        # print("\npast split tokens num:", len(past_split_tokens))
        # if len(past_split_tokens) > 0:
        #     print("\npast split tokens:", past_split_tokens)
        # print("\nsplit_texts:", split_texts)

        
        merge_split_texts = self.merge_past_now(past_split_tokens, split_texts)
        merge_split_texts = split_texts

        # print("\nMerge:\n", "".join(merge_split_texts))
        
        context = data_dict[self.config["data_format_map"]["context"]].strip()+"\n\n"
        question = data_dict[self.config["data_format_map"]["question"]]
        if len(question.strip()) > 0:
            question = question.strip()+"\n\n"
        else:
            question = "" # data 中没有 question, question需要从prompt template中额外获取

        input_ids, attention_mask, index_split = [], [], []
        question_start_index, question_end_index = 1<<28, 1<<28
        context_start_index, context_end_index = 1<<28, 1<<28, 
        prefix_same_tokens = []
        start = 0
        
        """
        目前这个写法要求在整个prompt中splite_sentence是能够将context, quesiton, prompt_template之间完全干净切开，如果context，quesiton中加入额外text token，就会出错。因此注意，在prompt_template中将context, question 与 pormpt 部分的text通过"\n\n"进行完全区分。
        另外就是在build_prompt与context,question部分进行拼接时，要context.strip(), question.strip()，即LongBenchV2中处理方式。
        
        目前这个写法在寻找context, question部分的segment index有个小问题：
        1. 当question_first or context_first 与 prompt_template中有相同的sentence。
        这种情况在实际出现的概率极低，暂时先不需要进行处理。
        """
        
        if question != "":
            question_splited = self.split_sentence(question, spliter=spliter)
            question_first, question_last = question_splited[0], question_splited[-1]
            question_token_length = sum([len(self.tokenizer.encode(text, add_special_tokens=False)) for text in question_splited]) 
       
        context_splited = self.split_sentence(context, spliter=spliter)
        context_first, context_last = context_splited[0], context_splited[-1]
        context_token_length = sum([len(self.tokenizer.encode(text, add_special_tokens=False)) for text in context_splited])
        
        past_sen_num = len(past_split_tokens)
        for index, text in enumerate(merge_split_texts):
            # 为了防止出现past_token_id != decode(encode(past_token_id))的情况，例如Llama3-8B，这种情况 text竟然还是一样的。
            if index < past_sen_num:
                token_ids = past_split_tokens[index]
                token_num = len(token_ids)
            else:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)  #  bos token
                token_num = len(token_ids)
            
            if start + token_num <= self.PREFIX_PROMPT_TOKEN_NUM:
                prefix_same_tokens.extend(token_ids)
            
            if index >= past_sen_num:
                input_ids.extend(token_ids)
            
            if start < context_start_index and index >= past_sen_num and \
                (text.strip() == context_first.strip() or text.replace("<text>\n\n", "").strip() == context_first.strip()):
                context_start_index = start
                context_end_index = context_start_index + context_token_length
            
            if question != "":
                if start < question_start_index and start >= past_token_num and \
                    (start < context_start_index or start >= context_end_index) and text.strip() == question_first.strip():
                    question_start_index = start - past_token_num
                    question_end_index = question_start_index + question_token_length
            
            index_split.append(list(range(start, start+token_num)))
            start += token_num
            attention_mask.extend([1] * token_num)
        
        assert self.verify_text_index(context, input_ids,
                                      context_start_index-past_token_num,
                                      context_end_index-past_token_num, spliter=spliter), \
            "Searching Context Index in Final Input's Texts(after building prompt) Error! \nSplited texts: " + str(merge_split_texts) + "Context First:" + context_first

        if question != "":
            assert self.verify_text_index(question, input_ids, 
                                          question_start_index, question_end_index, spliter=spliter), \
                "Searching Question Index in Final Input's Texts(after building prompt) Error! \nSplited texts: " + str(merge_split_texts) + "Question First:" + question_first
        else:
            question_start_index = context_end_index - past_token_num
            question_end_index = start - past_token_num
        
        if self.config["add_prompt"]:
            question_end_index = start - past_token_num

        model_inputs = {"input_ids": torch.tensor([input_ids], device=self.model.device), 
                        "attention_mask": torch.tensor([attention_mask], device=self.model.device)}
        
        process_info = {"merge_past_context": "".join(merge_split_texts),
                        "sentence_index_split": index_split,
                        "question_start_index": question_start_index,
                        "question_end_index": question_end_index,
                        "context_start_index": context_start_index,
                        "context_end_index": context_end_index,
                        "prefix_same_token_num": len(prefix_same_tokens),
                        "past_token_num": past_token_num}
    
        return model_inputs, process_info
            
    
    def select_sentence(self, selected_token_index, index_split):
        """
        Return all token index in sentences that selected according to the selected token index
        Complex: O(all input token)
        """
        sentence_num = len(index_split)
        token2sentence = {}
        for index_sen, split in enumerate(index_split):
            for index_id in split:
                token2sentence[index_id] = index_sen

        select_sentence_index = [0] * sentence_num
        for token_index in selected_token_index:
            select_sentence_index[token2sentence[token_index]] = 1
        
        cache_token_split_index = []
        for index in range(sentence_num):
            if select_sentence_index[index] == 1:
                cache_token_split_index.append(index_split[index])
        
        return cache_token_split_index
    
    
    def focus_key_sentence(self, data_dict, prompt, retri_split_pattern,
                           cache=None, past_split_tokens=[], phrase_token_num=10,
                           past_token_num=0, topk=10, doc_id=None):
        
        try:
            model_inputs, process_info = self.pre_process(
                data_dict=data_dict, prompt=prompt, 
                spliter=retri_split_pattern,
                past_split_tokens=past_split_tokens, 
                past_token_num=past_token_num
            )
        except AssertionError as e:
            print(e)
        
        with torch.no_grad():
            if cache is None:
                output = self.model(**model_inputs, output_attentions=True)
            else:
                # past_key_value_length = cache[0][0].shape[2]
                cache = DynamicCache.from_legacy_cache(cache)
                # legacy_format_cache = cache.to_legacy_cache()
                output = self.model(**model_inputs, past_key_values=cache, output_attentions=True)

        if cache is None:
            self.PREFIX_PROMPT_TOKEN_NUM = process_info["context_start_index"]
        else:
            assert self.PREFIX_PROMPT_TOKEN_NUM == process_info["prefix_same_token_num"], \
                "Prefix prompt token nummber:[%d] should equal with prefix_same_token_num:[%d] in each chunk pre_process." % \
                    (self.PREFIX_PROMPT_TOKEN_NUM, process_info["prefix_same_token_num"])
        
        focus_indexs, attention_weights, token_scores, q_w = self.thoughts.get_focus_index(
            PREFIX_PROMPT_TOKEN_NUM=self.PREFIX_PROMPT_TOKEN_NUM,
            process_info=process_info,
            output_attentions=output["attentions"],
            topk=topk,
            # computer_method = 1 if self.QUESTION_WEIGHT is False else 2,
            compute_method=self.compute_score_method,
            first_q_w=self.QUESTION_WEIGHT,
            phrase_token_num=phrase_token_num
        )

        if self.QUESTION_WEIGHT is None:
            if self.use_question_weight:
                self.QUESTION_WEIGHT = q_w
            else:
                self.QUESTION_WEIGHT = np.ones((process_info["question_end_index"] - \
                                                process_info["question_start_index"], 1))
        
        malplot_index = [index-self.PREFIX_PROMPT_TOKEN_NUM for index in focus_indexs]

        # extral add token index from where is prefix to prompt in each chunk process
        # use_past_key_value default is False, no use past_key_value, only retrieval context token
        if self.PREFIX_PROMPT_TOKEN_NUM > 0 and self.config["use_past_key_value"]:
            focus_indexs.extend([index for index in range(self.PREFIX_PROMPT_TOKEN_NUM)])

        select_split_indexs = self.select_sentence(focus_indexs, process_info["sentence_index_split"])
        # print("select_split_indexs:", select_split_indexs)
        select_indexs = [index for sublist in select_split_indexs for index in sublist]
        
        # select_split_tokens include prefix prompt and select token from context
        if len(past_split_tokens) > 0:
            past_token_ids = [token for sublist in past_split_tokens for token in sublist]
        else:
            past_token_ids = []
        complete_input_ids = past_token_ids + model_inputs["input_ids"][0].cpu().tolist()
        
        input_length = attention_weights.shape[1]
        assert len(complete_input_ids) == input_length, "%d != %d" % (input_length, len(complete_input_ids))

        if self.record_attention:
            if doc_id:
                dir_name = "_".join(doc_id.split("_")[:-1])
                record_dir = os.path.join(self.record_attention, dir_name)
            else:
                record_dir = self.record_attention

            if not os.path.exists(record_dir):
                os.makedirs(record_dir)
            
            x_tokens = complete_input_ids[self.PREFIX_PROMPT_TOKEN_NUM:\
                                          process_info["context_end_index"]] # key展示到context结尾
            x2_tokens = complete_input_ids[self.PREFIX_PROMPT_TOKEN_NUM:\
                                           process_info["question_end_index"]+past_token_num] # key展示到question结尾
            y_tokens = complete_input_ids[process_info["question_start_index"]+past_token_num:\
                                          process_info["question_end_index"]+past_token_num] 
                                          
            x_labels = [self.tokenizer.decode(id) for id in x_tokens]
            x2_labels = [self.tokenizer.decode(id) for id in x2_tokens]
            y_labels = [self.tokenizer.decode(id) for id in y_tokens]

            question2context = np.copy(
                attention_weights[process_info["question_start_index"]: process_info["question_end_index"], 
                                  self.PREFIX_PROMPT_TOKEN_NUM: process_info["context_end_index"]]
            )
            question2context_question = np.copy(
                attention_weights[process_info["question_start_index"]: process_info["question_end_index"], 
                                  self.PREFIX_PROMPT_TOKEN_NUM: process_info["question_end_index"]+past_token_num]
            )
            
            assert len(x_labels) == len(question2context[0]), "row token length must match"
            assert len(y_labels) == len(question2context), "col token length must match"

            # # 只画 question2context，即到context结尾
            # self.thoughts.malplot_attention(
            #     attention_weights=self.QUESTION_WEIGHT*self.QUESTION_WEIGHT*question2context,
            #     x_labels=x_labels,
            #     y_labels=y_labels,
            #     focus_indexs=malplot_index,
            #     save_name="%s/%s_attention.png" % (dir_name, doc_id),
            # )
            
            # 不仅画question2context, context部分一直到question结尾
            # self.thoughts.malplot_attention(
            #     attention_weights=self.QUESTION_WEIGHT*self.QUESTION_WEIGHT*question2context_question,
            #     x_labels=x2_labels,
            #     y_labels=y_labels,
            #     focus_indexs=malplot_index,
            #     save_name="%s/%s_attention2.png" % (dir_name, doc_id),
            # )
            
            # 先保存每个iteration attention weight, 最后全部画在一张图里
            self.thoughts.add_malplot_attention(
                attention_weights=self.QUESTION_WEIGHT*self.QUESTION_WEIGHT*question2context_question,
                x_labels=x2_labels,
                y_labels=y_labels,
                focus_indexs=malplot_index,
            )
            
            ## 画 token_scores 
            # assert len(x_labels) == len(token_scores), "row token length must match"
            # self.thoughts.malplot_attention(
            #     attention_weights=np.expand_dims(token_scores, axis=0),
            #     x_labels=x_labels,
            #     y_labels=["Question Sum"],
            #     focus_indexs=malplot_index,
            #     save_name="%s/%s_token_scores.png" % (dir_name, doc_id),
            # )
        
        # get prompt last text
        if self.LAST_PROMPT_TEXT is None:
            self.LAST_PROMPT_TEXT = self.tokenizer.decode(
                complete_input_ids[process_info["context_end_index"]:]
            )
        
        select_split_tokens = []
        for sentence_index in select_split_indexs:
            tmp = []
            for token_index in sentence_index:
                token_id = complete_input_ids[token_index]
                tmp.append(token_id)
            select_split_tokens.append(tmp)

        select_tokens = [token for sublist in select_split_tokens for token in sublist]
        focus_text = self.tokenizer.decode(select_tokens)
        print_info = {"doc_id": doc_id,
                      # "context": data_dict[self.config["data_format_map"]["context"]],
                      # "question": data_dict[self.config["data_format_map"]["question"]],
                      # "prompt": prompt,
                      # "merge_past_context": process_info["merge_past_context"],
                      # "compute_score_method": self.compute_score_method,
                      # "phrase_token_num": self.phrase_token_num,
                      # "question_weights": str(self.QUESTION_WEIGHT),
                      "focus_text": focus_text,
                      # "context_start_index": process_info["context_start_index"],
                      # "context_end_index": process_info["context_end_index"],
                      # "question_start_index": process_info["question_start_index"],
                      # "question_end_index": process_info["question_end_index"],
                      # "chunk_size": self.config["chunk_size"],
                      # "topk_num": topk,
                      "original_num": model_inputs["input_ids"].shape[1],
                      "select_num": len(select_tokens),
                      "cost_time": float(time.time()-self.START_TIME)}
            
        # return output["past_key_values"], select_indexs, select_split_tokens, print_info
        if self.config["use_past_key_value"]:
            return output, select_indexs, select_split_tokens, print_info
        else:
            del output
            torch.cuda.empty_cache()
            return None, select_indexs, select_split_tokens, print_info

