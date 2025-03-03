#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

from .pipline import Pipline

class InfiniRetri:
    def __init__(self, model=None, tokenizer=None, name_or_path=None,
                 topk=300, answer_length=8, window_length=1024):
        
        assert model != None or name_or_path != None, \
            "At least one of the two parameter 'model' or 'model_name_or_path' shoulde be assigned a value. "
        
        config = {"topk": topk,
                  "phrase_token_num": answer_length,
                  "chunk_size": window_length}
        
        if name_or_path != None:
            config["llm_name_or_path"] = name_or_path
            self.ir = Pipline(config=config)
        else:
            assert tokenizer != None, "The parameter of tokenizer should be assigned a value."
            self.ir = Pipline(model=model, tokenizer=tokenizer, config=config)

    def generate(self, context, question=None, prompt=None, topk=None, 
                 answer_length=None, window_length=None, temperature=0.5, 
                 retri_split_pattern=None, max_new_tokens=128, print_info=False):
    
        if question == None:
            question = context[-512:]
        
        if prompt == None:
            prompt = "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion:\n\n{question}\n\nAnswer:"
        
        data_dict = {"context": context, "question": question}
        response, info = self.ir.generate2(data_dict=data_dict, prompt=prompt, gen_name="InfiniRetri",
                                           chunk_size=window_length, topk=topk, phrase_token_num=answer_length,
                                           retri_split_pattern=retri_split_pattern, temperature=temperature, 
                                           max_new_tokens=max_new_tokens)
        
        if print_info == True:
            print(info)

        return response







