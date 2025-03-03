#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import os
import time

import numpy as np
from scipy.signal import convolve2d
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class thoughts:
    def __init__(self, config={}):
        self.cmap = config.get("cmap", "Reds")
        self.plt_save = config.get("plt_save", None)
        self.malplot_all_init()

    def malplot_all_init(self):
        self.all_attention_weights = []
        self.all_x_labels = []
        self.all_y_labels = []
        self.all_focus_indexs = []

    
    def softmax(self, matrix, axis=-1):
        exp_matrix = np.exp(matrix - np.max(matrix, axis=axis, keepdims=True))
        softmax_matrix = exp_matrix / np.sum(exp_matrix, axis=axis, keepdims=True)
        return softmax_matrix
        
    
    def compute_token_scores(self, attentions, context_segments, question_segments, past_token_num):
        
        layers = -1        # select attention weights (attention scores) from last layers
        select_attention = attentions[layers] # tensor shape: (1, head_num, len, len)
        attention_weights = select_attention[0].sum(axis=0)  # get sum from multi-head attention scores, tensor shape: (len, len)

        attention_weights = attention_weights.float().cpu().detach().numpy()
        
        # select token from question to context attention scores 
        # context start index default 0, shape:(question len, merge context len)
        context_indexs = []
        for segment in context_segments:
            context_indexs.extend(list(range(segment[0], segment[1])))
        context_col_select = np.r_[context_indexs]

        question_indexs = []
        for segment in question_segments:
            question_indexs.extend(list(range(segment[0], segment[1])))
        question_row_select = np.r_[question_indexs]
        question2context = np.copy(attention_weights[question_row_select, context_col_select]) 
                                                                           
        token_scores = np.sum(question2context, axis=0) # ndarray shape: (context len, 1)
        return token_scores, attention_weights, None

    
    def compute_token_scores2(self, attentions, context_segments, 
                              question_segments, past_token_num, first_q_w):
        
        layers = -1        # select attention weights (attention scores) from last layers
        select_attention = attentions[layers] # tensor shape: (1, head_num, len, len)
        attention_weights = select_attention[0].sum(axis=0)  # get sum from multi-head attention scores, tensor shape: (len, len)
        attention_weights = attention_weights.float().cpu().detach().numpy()
                 
        # select token from question to context attention scores 
        # context start index default 0, shape:(question len, merge context len)
        context_indexs = []
        for segment in context_segments:
            context_indexs.extend(listr(range(segment[0], segment[1])))
        context_col_select = np.r_[context_indexs]

        question_indexs = []
        for segment in question_segments:
            question_indexs.extend(list(range(segment[0], segment[1])))
        question_row_select = np.r_[question_indexs]
        question2context = np.copy(attention_weights[question_row_select, context_col_select]) 

        if first_q_w is None:
            question_indexs = []
            for segment in question_segments:
                question_indexs.extend(list(range(segment[0] + past_token_num, segment[1] + past_token_num)))
            question_col_select = np.r_[question_indexs]
            question2question = np.copy(attention_weights[question_row_select, question_col_select])
                                                
            question_weights = np.sum(question2question, axis=1, keepdims=True) # shape: (question len)
        else:
            question_weights = first_q_w

        question2context = question_weights*question2context
        question2context = question_weights*question2context
        # question2context = question_weights*question2context
        
        token_scores = np.sum(question2context, axis=0) # ndarray shape: (context len, 1)
        return token_scores, attention_weights, question_weights


    def compute_phrase_scores(self, attentions, context_segments, question_segments,
                              past_token_num, first_q_w, phrase_token_num=5):
                              

        """ convid. 
        """
        layers = -1        # select attention weights (attention scores) from last layers
        select_attention = attentions[layers] # tensor shape: (1, head_num, len, len)
        attention_weights = select_attention[0].sum(axis=0)  # get sum from multi-head attention scores, tensor shape: (len, len)

        attention_weights = attention_weights.float().cpu().detach().numpy()
        # print("attention weighst shape:", attention_weights.shape)
                 
        # select token from question to context attention scores 
        # context start index default 0, shape:(question len, merge context len)
        context_indexs = []
        for segment in context_segments:
            context_indexs.extend(list(range(segment[0], segment[1])))
        context_col_select = np.r_[context_indexs]

        question_indexs = []
        for segment in question_segments:
            question_indexs.extend(list(range(segment[0], segment[1])))
        question_row_select = np.r_[question_indexs]
        question2context = np.copy(attention_weights[question_row_select,:][:, context_col_select]) 
        
        # 当前位置向后计算phrase_token_num个位置分数
        # kernel = np.ones((1, phrase_token_num))
        # padded_array = np.pad(question2context, ((0, 0), (0, phrase_token_num-1)), \
        # 	                  mode='constant', constant_values=0)
        # question2context = convolve2d(padded_array, kernel, mode='valid')
        
        #当前位置向左右两侧计算phrase_token_num个位置，选token
        kernel = np.ones((1, phrase_token_num))
        question2context = convolve2d(question2context, kernel, mode="same", boundary="fill", fillvalue=0)
        
        if first_q_w is None:
            question_indexs = []
            for segment in question_segments:
                question_indexs.extend(list(range(segment[0] + past_token_num, segment[1] + past_token_num)))
            question_col_select = np.r_[question_indexs]
            question2question = np.copy(attention_weights[question_row_select,:][:, question_col_select])               
            question_weights = np.sum(question2question, axis=1, keepdims=True) # shape: (question len)
        else:
            question_weights = first_q_w

        question2context = question_weights*question2context
        question2context = question_weights*question2context
        
        token_scores = np.sum(question2context, axis=0) # ndarray shape: (context len, 1)
        return token_scores, attention_weights, question_weights


    def get_focus_index(self, PREFIX_PROMPT_TOKEN_NUM, 
                        process_info, output_attentions, topk, 
                        compute_method=2, first_q_w=None, phrase_token_num=5):

        computer_context_start = PREFIX_PROMPT_TOKEN_NUM # 0
        context_segments = [[computer_context_start, process_info["context_end_index"]]]

        # question_segments = [[0, process_info["context_start_index"]], 
        #                      [process_info["question_start_index"], process_info["question_end_index"]]]
        
        question_segments = [[process_info["question_start_index"], process_info["question_end_index"]]]

        if compute_method == 1:
            token_scores, attention_weights, q_w = self.compute_token_scores(
                output_attentions,
                context_segments,
                question_segments,
                process_info["past_token_num"]
            )
        elif compute_method == 2:
            token_scores, attention_weights, q_w = self.compute_token_scores2(
                output_attentions,
                context_segments,
                question_segments,
                process_info["past_token_num"],
                first_q_w=first_q_w
            )
        elif compute_method == 3:
            token_scores, attention_weights, q_w = self.compute_phrase_scores(
                output_attentions,
                context_segments,
                question_segments,
                process_info["past_token_num"],
                first_q_w=first_q_w,
                phrase_token_num=phrase_token_num
            )
        
        # all_token_num = output["attentions"].shape[-1]
        # topk = int(all_token_num * 0.01)
        
        topk_token_indices = np.argsort(token_scores[PREFIX_PROMPT_TOKEN_NUM-computer_context_start:])[-topk:] + PREFIX_PROMPT_TOKEN_NUM       
        return topk_token_indices.tolist(), attention_weights, token_scores, q_w


    def malplot_attention(self, attention_weights, 
                          x_labels=None, y_labels=None, 
                          focus_indexs=None, save_name=None, pic_size=358):

        row_size, col_size = attention_weights.shape[0], attention_weights.shape[1]
        aspect = col_size / row_size
        plt.figure(figsize=(pic_size, pic_size/aspect))

        if x_labels is None and y_labels is None:
            heatmap = sns.heatmap(attention_weights, fmt=".2f", cmap=self.cmap)
        else:
            heatmap = sns.heatmap(attention_weights, xticklabels=x_labels, 
                                  yticklabels=y_labels, fmt=".2f", cmap=self.cmap)
        
        # Marked the position of focus token index in attention weights headmap
        if focus_indexs is not None:
            for focus_index in focus_indexs:
                # The uppermost leftmost coordinate position of rectangle, rectangle width, rectagngle height
                rect = Rectangle((focus_index, 0), 1, row_size, fill=False, edgecolor='g', linewidth=1)
                heatmap.add_artist(rect)
        
        plt.title("Attention Weights Heatmap")
        plt.xlabel("Key Token")
        plt.ylabel("Query Token")
        
        if self.plt_save and save_name:
            save_path = self.plt_save + "/" + save_name
            heatmap.get_figure().savefig(save_path)
            # print("Save %s successfully!" % (save_path))
    

    def add_malplot_attention(self, attention_weights, x_labels, y_labels, focus_indexs):
        self.all_attention_weights.append(attention_weights)
        self.all_x_labels.append(x_labels)
        self.all_y_labels.append(y_labels)
        self.all_focus_indexs.append(focus_indexs)
    

    def malplot_all_attention(self, save_name, max_width=598, max_height=598, page_max_num=598):

        attention_num = len(self.all_attention_weights)
        page_num = (attention_num-1) // page_max_num + 1
        for page in range(page_num):          
            t1 = time.time()
            start_iter, end_iter = page*page_max_num, min((page+1)*page_max_num, attention_num)
            print("Starting to generate attention heatmap from %d to %d iteration......" % (start_iter, end_iter))
            
            data_matrices = self.all_attention_weights[start_iter:end_iter]
            max_cols = max(data.shape[1] for data in data_matrices)
            total_rows = sum(data.shape[0] for data in data_matrices)

            fig_width = min(max_cols, max_width)
            fig_height = min(total_rows, max_height)
            fig = plt.figure(figsize=(fig_width/5, fig_height/5))

            file_name = save_name.split("/")[0]
            fig_title = "%s's Intermediate Attention Heatmap — %d to %d Iteration" % (file_name, start_iter, end_iter)
            fig.suptitle(fig_title, fontsize=38, y=1.0, horizontalalignment="center")

            gs = gridspec.GridSpec(len(data_matrices), 1)
            for index, data in enumerate(data_matrices):
                ax = fig.add_subplot(gs[index])
                rows, cols = data.shape
                aspect = cols / rows
                ax.set_box_aspect(1/aspect)
                if self.all_x_labels is None or self.all_y_labels is None:
                    heatmap = sns.heatmap(data, ax=ax, cbar=True, square=True, cmap=self.cmap)
                else:
                    heatmap = sns.heatmap(data, ax=ax, cbar=True, square=True, cmap=self.cmap,
                        xticklabels=self.all_x_labels[start_iter+index], yticklabels=self.all_y_labels[start_iter+index])
                
                ax.set_title(f"Attention Weights Heatmap at %d Iteration" % (start_iter+index+1), fontsize=25)
                ax.set_xlabel(f"Key Token", fontsize=15)
                ax.set_ylabel(f"Query Token", fontsize=15)
                ax.tick_params(axis="both", which="major", labelsize=5)
                # ax.set_axis_off()
                
                # Marked the position of focus token index in attention weights headmap
                if self.all_focus_indexs is not None:
                    for focus_index in self.all_focus_indexs[start_iter+index]:
                        # The uppermost leftmost coordinate position of rectangle, rectangle width, rectagngle height
                        rect = Rectangle((focus_index, 0), 1, rows, fill=False, edgecolor='g', linewidth=1)
                        heatmap.add_artist(rect)

            # plt.subplots_adjust(hspace=0.9)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            if self.plt_save and save_name:
                save_path = self.plt_save + "/" + save_name + "attention_heatmap_page%d.png" % (page+1) 
                
                plt.savefig(save_path, bbox_inches="tight", dpi=258)
            print("Cost %fs to geneate all attention heatmap save in %s" % (time.time()-t1, save_path))

    
