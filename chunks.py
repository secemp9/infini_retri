#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import re
from typing import List, Optional, Tuple, Any

import tiktoken
from transformers import AutoTokenizer


class Chunks:
    def __init__(
            self,
            config: dict,
            tokenizer=None
    ):
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(config["llm_name_or_path"])
        else:
            self.tokenizer = tokenizer

        self.config = config
        # self.split_sen_pattern = self.config.get("retri_split_pattern", r"([。！？；.?!;])")
        self.split_sen_pattern = self.config.get("retri_split_pattern")
        
            
    # def split_sentence(self, text: str, spliter: str):
    #     """Split by punctuation from spliter and keep punctuation

    #     Args:
    #         text (str):  original input text
    #         spliter (str): spliter punctuation

    #     Returns:
    #         list: a collection of each individual sentence 
    #     """
   
    #     # text = text.strip() #
    #     sentence_list = re.split(spliter, text)

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
        Split by punctuation and keep punctuation
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
                # 处理当英文字母中名字姓氏之间的'.'不切分（少于5个字母不切）
                if len(cur_sen.strip()) == 0 \
                or len(sen.strip()) == 0 or sen.strip() in punctuation \
                or cur_sen[-1] == '.' and len(sen.strip()) <= 10:
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
    
    
    def split_into_chunks(self, text: str, chunk_size: int, spliter: str) -> list[str]:
        """ Split by punctuation from spliter and keep punctuation, 
        Rearrange sentences and punctuation.

        Args:
            text (str): the original input context that was splited
            chunk_size (int): size of each splited chunk 
            spliter (str, optional): the punctuation used to cut individual sentences in text. Defaults to r'([。！？；.?!;])'.

        Returns:
            list[str]: a collection of continuous text segements obtained by spliting text
        """
        sentences = self.split_sentence(text, spliter)

        chunks = []
        current_chunk = ""
        
        # 
        for sentence in sentences:
            sentence_length = self.get_prompt_length(sentence)

            if self.get_prompt_length(current_chunk) + sentence_length <= chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    if self.get_prompt_length(current_chunk) <= chunk_size:
                        chunks.append(current_chunk)
                    else:
                        if spliter != ' ':  # Avoid infinite loops
                            chunks.extend(self.split_into_chunks(
                                current_chunk, chunk_size=chunk_size, spliter=' '))
                current_chunk = sentence
        
        if current_chunk != '':
            if self.get_prompt_length(current_chunk) <= chunk_size:
                chunks.append(current_chunk)
            else:
                if spliter != ' ':  # Avoid infinite loops
                    chunks.extend(self.split_into_chunks(
                        current_chunk, chunk_size=chunk_size, spliter=' '))
        
        # Re-segment the last two blocks to make length of two be about equal
        if len(chunks) > 1 and self.get_prompt_length(chunks[-1]) < chunk_size//2:
            last_chunk = chunks.pop()
            penultimate_chunk = chunks.pop()
            combined_text = penultimate_chunk + last_chunk

            new_sentences = self.split_sentence(combined_text, spliter)

            # Reallocate sentence using double pointer
            new_penultimate_chunk = ""
            new_last_chunk = ""
            i, j = 0, len(new_sentences) - 1

            while i <= j and len(new_sentences) != 1:
                flag = False
                if self.get_prompt_length(new_penultimate_chunk + new_sentences[i]) <= chunk_size:
                    flag = True
                    new_penultimate_chunk += new_sentences[i]
                    if i == j:
                        break  
                    i += 1
                if self.get_prompt_length(new_last_chunk + new_sentences[j]) <= chunk_size:
                    new_last_chunk = new_sentences[j] + new_last_chunk
                    j -= 1
                    flag = True
                if flag == False:
                    break
            if i < j:
                # If there is any unallocated part, split it by punctuation or space and then allocate it
                remaining_sentences = new_sentences[i:j+1]
                if remaining_sentences:
                    remaining_text = "".join(remaining_sentences)
                    words = remaining_text.split(' ')
                    end_index = len(words)-1
                    for index, w in enumerate(words):
                        if self.get_prompt_length(' '.join([new_penultimate_chunk, w])) <= chunk_size:
                            new_penultimate_chunk = ' '.join(
                                [new_penultimate_chunk, w])
                        else:
                            end_index = index
                            break
                    if end_index != len(words)-1:
                        new_last_chunk = ' '.join(
                            words[end_index:]) + ' ' + new_last_chunk
                            
            if len(new_sentences) == 1:
                chunks.append(penultimate_chunk)
                chunks.append(last_chunk)
            else:
                chunks.append(new_penultimate_chunk)
                chunks.append(new_last_chunk)

        return chunks
    

    def chunk_docs(self, doc: str, chunk_size:int, 
                   separator='\n', chunk_overlap=0, 
                   question=None) -> list[str]:
        
        """Split the an entire long context into continuous fragments of text, each longer than chunk_size. 

        Args:
            doc (str): the original long context to be splited
            chunk_size (int): the length of each splited text
            separator (str, optional): The first split pruntucation. Defaults to '\n'.
            chunk_overlap (int, optional): Split into overlapping lengths between each piece of doc. Defaults to 0.
            question (_type_, optional): question about doc. Defaults to None.

        Returns:
            list[str]: a collection of continuous text segements obtained by spliting doc 
        """
    
        if question != None:
            chunk_size = chunk_size - self.get_prompt_length(question)
            
        # splits = doc.split(separator)
        splits = self.split_sentence(doc, self.split_sen_pattern)
        splits = [s for s in splits if s != ""]
        separator_len = self.get_prompt_length_no_special(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            now_len = self.get_prompt_length_no_special(d)
            if (
                total + now_len + (separator_len if len(current_doc) > 0 else 0)
                > chunk_size
            ):
                if total > chunk_size:
                    # print(
                    #     f"Created a chunk of size {total}, "
                    #     f"which is longer than the specified {chunk_size}"
                    # )

                    if len(current_doc) == 1:  # if one chunk is too long
                        split_again = self.split_into_chunks(
                            current_doc[0], chunk_size, self.split_sen_pattern)
                        docs.extend(split_again)
                        current_doc = []
                        total = 0

                if len(current_doc) > 0:
                    doc = ''.join(current_doc)
                    # doc = separator.join(current_doc)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > chunk_overlap or (
                        total + now_len +
                            (separator_len if len(current_doc) > 0 else 0)
                        > chunk_size
                        and total > 0
                    ):
                        total -= self.get_prompt_length_no_special(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]

            current_doc.append(d)
            total += now_len + (separator_len if len(current_doc) > 1 else 0)

        # Check if the last one exceeds
        if self.get_prompt_length_no_special(current_doc[-1]) > chunk_size and len(current_doc) == 1:
            split_again = self.split_into_chunks(current_doc[0], chunk_size, self.split_sen_pattern)
            docs.extend(split_again)
            current_doc = []
        else:
            doc = ''.join(current_doc)
            # doc = separator.join(current_doc)
            if doc is not None:
                docs.append(doc)
        docs = [d for d in docs if d.strip() != ""]
        return docs


    def get_prompt_length(self, prompt, **kwargs: Any) -> int:
        if isinstance(prompt, list):
            prompt = self.join_docs(prompt)
        return len(self.tokenizer.encode(prompt, **kwargs))


    def get_prompt_length_no_special(self, prompt, **kwargs: Any) -> int:
        if isinstance(prompt, list):
            prompt = self.join_docs(prompt)
        if not isinstance(self.tokenizer, tiktoken.core.Encoding):
            return len(self.tokenizer.encode(prompt, add_special_tokens=False, **kwargs))
        else:
            return len(self.tokenizer.encode(prompt, disallowed_special='all', ** kwargs))


    def join_docs(self, docs: list[str]) -> str:
        if isinstance(docs, str):
            return docs
        return '\n\n'.join(docs)
        # return ''.join(docs)
    
