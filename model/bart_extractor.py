import os
import sys
import math
import torch
import torch.nn as nn
from transformers import RobertaModel,RobertaForTokenClassification
from transformers import RobertaTokenizer
import torch.nn.functional as F
import copy
import transformers
import numpy as np

class bart_extractor(nn.Module):

    def __init__(self, config, load_path=''):

        super(bart_extractor,self).__init__()  
        
        self.config=config
        
        seed = self.config.seed
        torch.manual_seed(seed)            
        torch.cuda.manual_seed(seed)       
        torch.cuda.manual_seed_all(seed)           
 
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.pretrained_model)
        if load_path=='':
            self.extractor = RobertaForTokenClassification.from_pretrained(self.config.pretrained_model,num_labels=1)
        else:
            self.extractor = RobertaForTokenClassification.from_pretrained(self.config.pretrained_model,num_labels=1,gradient_checkpointing=True)
            self.extractor.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage),strict=True)


    def forward(self, input_ids, input_mask, clss, mask_clss, return_focus=0):
        
        #input_id: (batch_size, sequence_length)   
        output = self.extractor(input_ids=input_ids,attention_mask=input_mask,output_attentions=True,output_hidden_states=True)
        retriever_all_logits=output.logits
        retriever_all_logits = retriever_all_logits.squeeze(2)
        sent_scores = retriever_all_logits.contiguous().view(-1)[clss.squeeze(0).cpu().tolist()].unsqueeze(0)  # shape = (1, n_turns)
        sent_scores = sent_scores * mask_clss.float()
        #sent_scores = nn.Sigmoid(sent_scores,1)
        
        
        #attention
        if self.config.attention_rollout == 1:
            attention = output.attentions  #tuple len=12, for each [1,head_num,token_length,token_length]
            top_attention = attention[:]
            top_attention = torch.cat(top_attention, 0)  #[layer_num,head_num,token_length,token_length]
            top_attention = torch.mean(top_attention, 1)
            top_attention=self.compute_joint_attention(top_attention)
            cls_top_attention=top_attention[:,clss,:].squeeze(1)  #[layer_num,cls_num,token_length]      
            cls_top_attention=torch.transpose(cls_top_attention, 0, 1)  #[cls_num,layer_num,token_length]
            #cls_top_attention=cls_top_attention.reshape(cls_top_attention.size(0),-1,cls_top_attention.size(2))
            
            #last_layer
            if self.config.attention_rollout_layer == 'last':
                cls_top_attention=cls_top_attention[:,-1:,:]
            else:
                pass
            
            cls_top_attention_all=torch.sum(cls_top_attention,1)
        else:
            attention = output.attentions  #tuple len=12, for each [1,head_num,token_length,token_length]
            top_attention = attention[:]
            top_attention = torch.cat(top_attention, 0)  #[layer_num,head_num,token_length,token_length]
            cls_top_attention=top_attention[:,:,clss,:].squeeze(2)  #[layer_num,head_num,cls_num,token_length]      
            cls_top_attention=torch.transpose(cls_top_attention, 0, 2)  #[cls_num,head_num,layer_num,token_length]
            cls_top_attention=cls_top_attention.reshape(cls_top_attention.size(0),-1,cls_top_attention.size(3))
            cls_top_attention_all=torch.sum(cls_top_attention,1)     
        
        #++++++++++++local context informatiion+++++++++
        #mask for bos and eos
        bos_mask=[]
        for i in input_ids:
            one_bos_mask=[]
            for tid,t in enumerate(i):
                if t == 0 or t == 2:
                    one_bos_mask.append(0)
                else:
                    one_bos_mask.append(1)
            bos_mask.append(one_bos_mask)    
        bos_mask=torch.tensor(bos_mask).cuda()

        cls_top_attention_all=cls_top_attention_all*bos_mask
        
        #sent-level split 
        stps=[]
        for i in input_ids:
            one_stps=[]
            for tid,t in enumerate(i):
                if t == 479:
                    one_stps.append(tid)
            stps.append(one_stps)
            
        #window-level split         
        window_indice=[]
        for idx, i in enumerate(input_ids):
            query_end=0
            for t in range(len(i)):
                if i[t] == self.config.eos_token_id:
                    query_end=t
                    break
            one_indice=list(range(query_end, len(i)))
            window_indice.append(one_indice)
        '''
        #sent split
        tar_idx=stps
        
        sent_to_id_list=[]
        for ss in tar_idx:
            sent_to_id={}
            s=ss
            for cls_id,cls_pos in enumerate(s):
                if cls_id >= len(s)-1:
                    cls_pos_next=-1
                else:
                    cls_pos_next=s[cls_id+1]
                
                sent_to_id[cls_id]=(cls_pos+1,cls_pos_next+1)
            sent_to_id_list.append(sent_to_id)
        '''

        #window split
        tar_idx=window_indice
        window=self.config.local_window_size
        sent_to_id_list=[]
        for sid,ss in enumerate(tar_idx):
            sent_to_id={}
            s=ss
            for cls_id,cls_pos in enumerate(s):
                if cls_pos + window >= s[-1]:
                    cls_pos_next=s[-1]
                else:
                    cls_pos_next=cls_pos + window
                
                sent_to_id[cls_id]=(cls_pos,cls_pos_next)
            sent_to_id_list.append(sent_to_id)


        focus_sent_batch=[]
        for s in sent_to_id_list:
            atten_on_sent=[]
            for k in s.keys():
                sent_k_atten=cls_top_attention_all[:, s[k][0]:s[k][1]]
                sent_k_atten_mean=torch.sum(sent_k_atten, 1)
                atten_on_sent.append(sent_k_atten_mean)
            atten_on_sent=torch.stack(atten_on_sent).T

            focus_sent_all=[]
            top=torch.topk(atten_on_sent, 30, dim=1).indices 
            
            for i in top:
                focus_sent_one_tar=[]
                occupy=[]
                
                for j in i.tolist():
                    start,end=s[j]
                    
                    tag=0
                    for k in occupy:
                        if start in range(k[0],k[1]) or end in range(k[0],k[1]):
                            tag=1
                    if tag==1:
                        continue
                            
                    
                    t=input_ids[:,start:end].squeeze(0)
                    focus_sent_one_tar.append(self.tokenizer.decode(t))
                    occupy.append((start,end))
                    if len(focus_sent_one_tar) == 5:
                        break
                    
                focus_sent_one_tar=[x.replace('</s><s>','') for x in focus_sent_one_tar]
                focus_sent_all.append(focus_sent_one_tar)
            focus_sent_batch.append(focus_sent_all)

        #++++++++++++global context informatiion+++++++++       
        #output hidden state
        hidden_state=output.hidden_states[-1]
        hidden_state=hidden_state[:,clss,:]
        hidden_state=hidden_state.squeeze(0)
        
        
        if return_focus==0:
            return sent_scores 
        else:
            return sent_scores,focus_sent_batch,hidden_state

    
    def compute_joint_attention(self, att_mat, add_residual=True):
        att_mat=att_mat.cpu()
        if add_residual:
            residual_att = np.eye(att_mat.shape[1])[None,...]
            aug_att_mat = att_mat + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[...,None]
        else:
            aug_att_mat =  att_mat
            
        aug_att_mat=torch.tensor(np.array(aug_att_mat)).cuda()
        joint_attentions = torch.tensor(np.zeros(aug_att_mat.shape)).cuda()
    
        layers = joint_attentions.shape[0]
        joint_attentions[0] = aug_att_mat[0]
        for i in np.arange(1,layers):
            #joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i-1])
            joint_attentions[i] = torch.mm(aug_att_mat[i],joint_attentions[i-1])
            
        return joint_attentions