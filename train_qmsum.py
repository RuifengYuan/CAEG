import time,copy
import argparse
import math
import torch
import torch.nn as nn
from rouge import Rouge
from transformers import RobertaTokenizer,BartTokenizer
from transformers import AdamW
from torch.optim import *
import numpy as np
from model.bart_extractor import *
from model.bart_generator import *
from model.dynamic_rag import *
from data_loader import data_loader
import os 
from tqdm import tqdm
import copy

def data_loader_from_ext_to_abs(text, summary, max_source,max_target, tokenizer, pad_token_id, query='', prompt=[],global_prompt=[],global_sent=[], max_sent=25, context='c'):
    '''
    text: list [sent1,sent2,..,sentn]
    summary,query: str
    prompt: list [[p1,p2,p3],[...],[...],[...]]
    '''
    context_input_ids = []
    context_attention_mask = []
    labels = None
    padded_text = text + max(max_sent-len(text),0)*['']
    global_sent = global_sent + max(max_sent-len(text),0)*['']
    prompt = prompt + max(max_sent-len(text),0)*['']
    global_prompt = global_prompt + max(max_sent-len(text),0)*['']
    
    for turn_id in range(max_sent):
        
        contextualized_turn = tokenizer.eos_token.join(padded_text[turn_id: turn_id + 1])
        global_turn = tokenizer.eos_token.join(global_sent[turn_id: turn_id + 1])

        if contextualized_turn != '':
            
            
            if context == 'c':
                src_texts=query + " // " + contextualized_turn
            if context == 'c+l':
                src_texts=query + " // " + " || ".join(prompt[turn_id]) + " // " + contextualized_turn                
            if context == 'c+l+g':
                src_texts=query + " // " + " || ".join(prompt[turn_id]) + " // " + " \\ ".join(global_prompt[turn_id]) + " // "  + contextualized_turn
            if context == 'c+g':
                src_texts=query + " // " + " || ".join(global_prompt[turn_id]) + " // " + contextualized_turn               
            
            input_dict = tokenizer.prepare_seq2seq_batch(src_texts= src_texts,
                                                                        tgt_texts=summary,
                                                                        max_length=max_source,
                                                                        max_target_length=max_target,
                                                                        padding="longest",
                                                                        truncation=True,
                                                                        )
        else:
            input_dict = tokenizer.prepare_seq2seq_batch(src_texts='',
                                                        tgt_texts=summary,
                                                        max_length=max_source,
                                                        max_target_length=max_target,
                                                        padding="longest",
                                                        truncation=True,
                                                        )            
        context_attention_mask.append(input_dict.attention_mask)
        context_input_ids.append(input_dict.input_ids)
        if labels is None:
            labels = input_dict.labels
        else:
            assert labels == input_dict.labels, '{} != {}'.format(labels, input_dict.labels)
    
    #context_input_ids = torch.LongTensor(context_input_ids).cuda()
    #context_attention_mask = torch.LongTensor(context_attention_mask).cuda()
    context_input_ids,context_attention_mask=pad_with_mask(context_input_ids)
    labels = torch.LongTensor([labels])[:max_target].cuda()
    
    return context_input_ids,context_attention_mask,labels

def pad_with_mask(data, pad_id=0, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)
        
    rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
    
    pad_mask = [[1] * len(d) + [0] * (width - len(d)) for d in data]
    return torch.LongTensor(rtn_data).cuda(),torch.LongTensor(pad_mask).cuda()

def cos_retrieval(target, target_idx, data_matrix, select_number):
    #target: tensor: 1xn
    #data_matrix: tensor: mxn
    cos=torch.nn.CosineSimilarity()
    
    score=cos(target,data_matrix)
    rank=torch.topk(score, select_number+1).indices
    
    re_rank=[]
    for i in rank.tolist():
        if i != target_idx:
            re_rank.append(i)
        if len(re_rank) == select_number:
            break
    
    return re_rank
    
    


class Train(object):

    def __init__(self, config):
        self.config = config  
        
        seed = self.config.seed
        torch.manual_seed(seed)            
        torch.cuda.manual_seed_all(seed)          
        

        self.tokenizer_ext = RobertaTokenizer.from_pretrained(self.config.pretrained_model_ext)
        self.tokenizer_abs = BartTokenizer.from_pretrained(self.config.pretrained_model_abs)
            
        self.log = open('log.txt','w')
        
        self.dataloader=data_loader('train', self.config, self.tokenizer_abs, 'abstractive', load_qmsum=1)

        self.extractor=bart_extractor(self.config, 'save_model/best-retriever.ckpt')   
        self.extractor.cuda()



        if self.config.mid_start == 0:
            self.generator = DynamicRagForGeneration.from_pretrained(self.config.pretrained_model_abs,
                                                                     n_docs=self.config.ndoc,
                                                                     gradient_checkpointing=True)
        else:    
            self.generator = DynamicRagForGeneration.from_pretrained(self.config.pretrained_model_abs,
                                                                     n_docs=self.config.ndoc,
                                                                     gradient_checkpointing=True)
            
            self.generator.load_state_dict(torch.load('save_model/best-generator.ckpt', map_location=lambda storage, loc: storage),strict=True)


        if self.config.multi_gpu==1:
            gpus=[int(gpu) for gpu in self.config.multi_device.split(',')]
            self.generator = nn.DataParallel(self.generator,device_ids=gpus, output_device=gpus[0])
            self.generator.cuda()
        else:
            self.generator.cuda()  


        param_optimizer = list(self.generator.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer], 'weight_decay': 0.01}]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.lr)        

        if self.config.use_lr_decay == 1:
            scheduler=lr_scheduler.StepLR(self.optimizer,step_size=1,gamma = self.config.lr_decay)

        
    def save_model(self, running_avg_loss,loss_list,rouge1,rouge2,loss_text=''):

        state = {
            'iter': self.dataloader.count/self.config.true_batch_size*self.config.batch_size,
            'ecop': self.dataloader.epoch,
            'generator':self.generator,
            'current_loss': running_avg_loss,
            'loss_list': loss_list,
            'rouge1':rouge1,
            'rouge2':rouge2,
            'config':self.config
        }
        try:
            model_save_path = self.config.save_path+str(self.dataloader.count/self.config.true_batch_size*self.config.batch_size)+'_iter_'+str(self.dataloader.epoch) +'_epoch__rouge_'+str(rouge1)[:6]+'_'+str(rouge2)[:6]+'__loss_'+str(running_avg_loss)[:6]+loss_text
            torch.save(state, model_save_path)
        except:
            print('can not save the model!!!')
        
        
        
    def train_one_batch(self):

        try:
            data_point = \
            self.dataloader.load_data()
        except:
            print('load data fail during the evaluation')
            return 0,0
            
        
        all_score=[]
        all_source=[]
        all_label=[]
        all_focus=[]
        all_hidden=[]

        for split in data_point:                
            batch_source_id,batch_source_id_mask,batch_clss,batch_clss_mask,batch_label,batch_weight,batch_source,batch_query,batch_summary = split
            try:
                with torch.no_grad():
                    output,focus_sent_batch,sent_hidden = self.extractor(batch_source_id,batch_source_id_mask,batch_clss,batch_clss_mask,return_focus=1)
                    sigmoid = nn.Sigmoid()
                    predicts_score=sigmoid(output)*batch_clss_mask     
                    
                all_score.append(predicts_score)
                all_label.append(batch_label)
                all_hidden.append(sent_hidden.squeeze(0))
                all_source+=batch_source[0]
                all_focus+=focus_sent_batch[0]
            except:
                continue


        all_score=torch.cat(all_score,1)    
        all_score=torch.squeeze(all_score)
        
        all_hidden=torch.cat(all_hidden,0)    
        
        all_label=torch.cat(all_label,1)    
        all_label=torch.squeeze(all_label)
        
        
        try:
            select_number=int(torch.sum(all_label))
            gold_top=torch.topk(all_label, select_number).indices
            pred_top=torch.topk(all_score, self.config.ndoc).indices         
            gold_top=gold_top.tolist()
            pred_top=pred_top.tolist()
        except:
            print('sent number is too small')
            return 0,0
        
        source=[]
        for idx in gold_top:
            source.append(all_source[idx])
        for idx in pred_top:
            if idx not in gold_top:
                source.append(all_source[idx])   

        source_score=[]
        for idx in gold_top:
            source_score.append(all_score[idx])
        for idx in pred_top:
            if idx not in gold_top:
                source_score.append(all_score[idx])     
            
        prompt=[]
        for idx in gold_top:
            prompt.append(all_focus[idx])
        for idx in pred_top:
            if idx not in gold_top:
                prompt.append(all_focus[idx])      
            
        hidden=[]
        for idx in gold_top:
            hidden.append((all_hidden[idx],idx))
        for idx in pred_top:
            if idx not in gold_top:
                hidden.append((all_hidden[idx],idx))
            
        '''
        global_prompt=[]
        global_sent=[]
        global_score=[]
        for idx,sent_and_idx in enumerate(hidden):
            sent,indice=sent_and_idx
            similar_sent=cos_retrieval(sent.unsqueeze(0),indice,all_hidden,3)
            
            global_prompt.append(all_focus[similar_sent[0]])
            global_sent.append(all_source[similar_sent[0]])
            global_score.append(all_score[similar_sent[0]])
        '''
        global_prompt=[]
        global_sent=[]
        global_score=[]
        for idx,sent_and_idx in enumerate(hidden[:self.config.ndoc]):
            sent,indice=sent_and_idx
            
            hidden_tensor=torch.cat([x[0].unsqueeze(0) for x in hidden],0)
            similar_sent=cos_retrieval(sent.unsqueeze(0),idx,hidden_tensor,3)
            
            global_prompt.append(prompt[similar_sent[0]])
            global_sent.append(source[similar_sent[0]])
            global_score.append(source_score[similar_sent[0]])
            
        
        topk=self.config.ndoc
        source=source[:topk]
        source_score=source_score[:topk]   
        prompt=prompt[:topk]
        global_prompt=global_prompt[:topk]
        global_sent=global_sent[:topk]
        global_score=global_score[:topk]
        
        
        if self.config.add_global_score == 1:
            for i in range(len(source_score)):
                source_score[i]+=global_score[i]
            
        
        
        source_score=torch.tensor([source_score]).cuda()
        source_score=torch.softmax(source_score,1)
        
        #source=[source]
        
        summary=data_point[0][8][0]
        query=data_point[0][7][0]
        
        prompt=[x[:self.config.local_window_number] for x in prompt]
        global_prompt=[x[:self.config.global_window_number] for x in global_prompt]
        
        context_input_ids,context_attention_mask,labels=data_loader_from_ext_to_abs(source,summary,self.config.max_article,self.config.max_summary,self.tokenizer_abs, self.config.pad_token_id, query, prompt, global_prompt, global_sent,max_sent=self.config.ndoc,context=self.config.context)       
      
        generator_outputs = self.generator(context_input_ids=context_input_ids,
                                           context_attention_mask=context_attention_mask,
                                           doc_scores=source_score,
                                           labels=labels)
        seq_loss = generator_outputs.loss
        seq_loss.backward()
        return seq_loss.item(), 1

    
 
    def train_iter(self):
        loss_list=[]

        count=0
        self.generator.train()
        for i in range(self.config.max_epoch*self.config.train_set_len):
            count=count+1
            time_start=time.time()
            
            success=0
            for j in range(int(self.config.true_batch_size/self.config.batch_size)):     
                loss,tag = self.train_one_batch()
                if tag == 1:
                    loss_list.append(loss)
                    success=success+1
                    
                if tag == 0:
                    print('one mini batch fail')                            
                    continue
                
            if success == int(self.config.true_batch_size/self.config.batch_size):                
                self.optimizer.step()                         
                self.optimizer.zero_grad()

                if self.config.use_lr_decay == 1:
                    if count%self.config.lr_decay_step == 0:
                        self.scheduler.step()         
            else:
                print('jump one batch')     
                
            time_end=time.time()                
            
            def loss_record(loss_list,window):
                recent_list=loss_list[max(0,len(loss_list)-window*int(self.config.true_batch_size/self.config.batch_size)):]
                return str(np.mean(recent_list))[:4]
            
            if count % self.config.checkfreq == 0:       
                record=str(count)+' iter '+str(self.dataloader.epoch) +' epoch avg_loss:'+loss_record(loss_list,100)
                    
                record+=' -- use time:'+str(time_end-time_start)[:5]
                print(record)
                
                
            if count % self.config.savefreq == 0 and count > self.config.savefreq-100 and count > self.config.startfreq:     
                recent_loss=loss_list[max(0,len(loss_list)-100*int(self.config.true_batch_size/self.config.batch_size)):]
                avg_loss=sum(recent_loss)/len(recent_loss)
                 
                print('start val')
                rouge1,rouge2=self.do_val(280)  
                print(rouge1,rouge2)
                
                self.save_model(avg_loss,loss_list,rouge1,rouge2) 
                self.generator.train()
                self.extractor.train()           
                
    def do_val(self, val_num):

        self.raw_rouge=Rouge()
        self.generator.eval()
        self.extractor.eval()

        data_loader_val=data_loader('val', self.config, self.tokenizer_ext, 'abstractive', load_qmsum=1)
     
        r1=[]
        r2=[]
                
        for i in tqdm(range(int(val_num)), desc='Validation'):       

            try:
                data_point = \
                data_loader_val.load_data()
            except:
                print('load data fail during the evaluation')
                return 0,0
            
        
            all_score=[]
            all_source=[]
            all_label=[]
            all_focus=[]
            all_hidden=[]
            for split in data_point:                
                batch_source_id,batch_source_id_mask,batch_clss,batch_clss_mask,batch_label,batch_weight,batch_source,batch_query,batch_summary = split
                try:
                    with torch.no_grad():
                        output,focus_sent_batch,sent_hidden = self.extractor(batch_source_id,batch_source_id_mask,batch_clss,batch_clss_mask,return_focus=1)
                        sigmoid = nn.Sigmoid()
                        predicts_score=sigmoid(output)*batch_clss_mask     
                        
                    all_score.append(predicts_score)
                    all_label.append(batch_label)
                    all_hidden.append(sent_hidden.squeeze(0))
                    all_source+=batch_source[0]
                    all_focus+=focus_sent_batch[0]
                except:
                    continue
    
    
            all_score=torch.cat(all_score,1)    
            all_score=torch.squeeze(all_score)
            
            all_hidden=torch.cat(all_hidden,0)    
            
            all_label=torch.cat(all_label,1)    
            all_label=torch.squeeze(all_label)
            
            #score argumentation by cos similarity 
            #all_score=score_argumentation(all_score, all_hidden, 0.25)
            
            
            select_number=int(torch.sum(all_label))
            gold_top=torch.topk(all_label, select_number).indices
            pred_top=torch.topk(all_score, self.config.ndoc).indices         
            gold_top=gold_top.tolist()
            pred_top=pred_top.tolist()
            
            source=[]
            #for idx in gold_top:
                #source.append(all_source[idx])
            for idx in pred_top:
                source.append(all_source[idx])   
    
            source_score=[]
            #for idx in gold_top:
                #source_score.append(all_score[idx])
            for idx in pred_top:
                source_score.append(all_score[idx])     
                
            prompt=[]
            #for idx in gold_top:
                #prompt.append(all_focus[idx])
            for idx in pred_top:
                prompt.append(all_focus[idx])      
                
            hidden=[]
            #for idx in gold_top:
                #hidden.append((all_hidden[idx],idx))
            for idx in pred_top:
                hidden.append((all_hidden[idx],idx))
                
            '''
            global_prompt=[]
            global_sent=[]
            global_score=[]
            for idx,sent_and_idx in enumerate(hidden):
                sent,indice=sent_and_idx
                similar_sent=cos_retrieval(sent.unsqueeze(0),indice,all_hidden,3)
                
                global_prompt.append(all_focus[similar_sent[0]])
                global_sent.append(all_source[similar_sent[0]])
                global_score.append(all_score[similar_sent[0]])
            '''
            global_prompt=[]
            global_sent=[]
            global_score=[]
            for idx,sent_and_idx in enumerate(hidden[:self.config.ndoc]):
                sent,indice=sent_and_idx
                
                hidden_tensor=torch.cat([x[0].unsqueeze(0) for x in hidden],0)
                similar_sent=cos_retrieval(sent.unsqueeze(0),idx,hidden_tensor,3)
                
                global_prompt.append(prompt[similar_sent[0]])
                global_sent.append(source[similar_sent[0]])
                global_score.append(source_score[similar_sent[0]])
                
            topk=self.config.ndoc
            source=source[:topk]
            source_score=source_score[:topk]   
            prompt=prompt[:topk]
            global_prompt=global_prompt[:topk]
            global_sent=global_sent[:topk]
            global_score=global_score[:topk]
            
            
            if self.config.add_global_score == 1:
                for i in range(len(source_score)):
                    source_score[i]+=global_score[i]
            source_score=torch.tensor([source_score]).cuda()
            source_score=torch.softmax(source_score,1)
            
            #source=[source]
            
            summary=data_point[0][8][0]
            query=data_point[0][7][0]
            
            with torch.no_grad():                        
                
                prompt=[x[:self.config.local_window_number] for x in prompt]
                global_prompt=[x[:self.config.global_window_number] for x in global_prompt]
                
                context_input_ids,context_attention_mask,labels=data_loader_from_ext_to_abs(source,summary,self.config.max_article,self.config.max_summary,self.tokenizer_abs, self.config.pad_token_id, query, prompt, global_prompt, global_sent,max_sent=self.config.ndoc,context=self.config.context)       
              
                outputs = self.generator.generate(context_input_ids=context_input_ids,
                                                   context_attention_mask=context_attention_mask,
                                                   doc_scores=source_score,
                                                   num_beams=1,
                                                   min_length=self.config.min_dec_steps,
                                                   max_length=self.config.max_dec_steps,
                                                   no_repeat_ngram_size=2,
                                                   length_penalty=1)
                
    
                assert isinstance(outputs, torch.Tensor)
                assert outputs.shape[0] == 1
                decoded_pred = self.tokenizer_abs.batch_decode(outputs, skip_special_tokens=True)
    
            pred=decoded_pred[0]    
            
            gold=summary
             
            scores = self.raw_rouge.get_scores(pred, gold)
            r1.append(scores[0]['rouge-1']['f'])
            r2.append(scores[0]['rouge-2']['f'])    

            
            if data_loader_val.epoch == 2:
                break
                    
        if len(r1) != 0 and len(r2) != 0:
            print(np.mean(r1),np.mean(r2))
            return np.mean(r1),np.mean(r2)
        else:
            return 0,0               



class Test(object):
    
    def __init__(self, config):
        self.config = config 

        self.config.seed=10   
        if 'ckpt' in self.config.test_model:
            self.generator = DynamicRagForGeneration.from_pretrained(self.config.pretrained_model_abs,
                                                                     n_docs=self.config.ndoc,
                                                                     gradient_checkpointing=True)
            self.generator.load_state_dict(torch.load('save_model/'+config.test_model, map_location=lambda storage, loc: storage),strict=True)
        else:
            x=torch.load('save_model/'+config.test_model,map_location='cpu')
            self.generator = x['generator']  
        
        
        self.generator.cuda()

        self.extractor=bart_extractor(self.config, 'save_model/best-retriever.ckpt')   
        self.extractor.cuda()
        
        self.tokenizer_ext = RobertaTokenizer.from_pretrained(self.config.pretrained_model_ext)
        self.tokenizer_abs = BartTokenizer.from_pretrained(self.config.pretrained_model_abs)
        
        self.raw_rouge=Rouge()
        
        import os 
        
        dir_path=config.test_model.split('epoch')[0]
        
        os.makedirs('result/'+dir_path)
        
        self.can_path = 'result/'+dir_path+'/'+config.test_model+'_cand.txt'

        self.gold_path ='result/'+dir_path+'/'+config.test_model+'_gold.txt'
        
        self.source_path ='result/'+dir_path+'/'+config.test_model+'_source.txt'


        
    
    def test(self,test_num=281):
        self.raw_rouge=Rouge()
        self.generator.eval()
        self.extractor.eval()
        
        data_loader_val=data_loader('test', self.config, self.tokenizer_ext, 'abstractive', load_qmsum=1)
     
        r1=[]
        r2=[]
        f1=[]
        f2=[]
        recall_all=[]
        
        r2_source=[]
        length=[]
        
        pred_list=[]
        gold_list=[]
        source_list=[]
        with open(self.can_path, 'w', encoding='utf-8') as save_pred:
            with open(self.gold_path, 'w', encoding='utf-8') as save_gold:
                with open(self.source_path, 'w', encoding='utf-8') as save_source:

                    for i in tqdm(range(int(test_num)), desc='testing'):            
                        
                        try:
                            data_point = \
                            data_loader_val.load_data()
                        except:
                            print('load data fail during the evaluation')
                            return 0,0
                        
                        all_score=[]
                        all_source=[]
                        all_label=[]
                        all_focus=[]
                        all_hidden=[]
                
                        for split in data_point:                
                            batch_source_id,batch_source_id_mask,batch_clss,batch_clss_mask,batch_label,batch_weight,batch_source,batch_query,batch_summary = split
                            try:
                                with torch.no_grad():
                                    output,focus_sent_batch,sent_hidden = self.extractor(batch_source_id,batch_source_id_mask,batch_clss,batch_clss_mask,return_focus=1)
                                    sigmoid = nn.Sigmoid()
                                    predicts_score=sigmoid(output)*batch_clss_mask     
                                    
                                all_score.append(predicts_score)
                                all_label.append(batch_label)
                                all_hidden.append(sent_hidden.squeeze(0))
                                all_source+=batch_source[0]
                                all_focus+=focus_sent_batch[0]
                            except:
                                continue
                
                
                        all_score=torch.cat(all_score,1)    
                        all_score=torch.squeeze(all_score)
                        
                        all_hidden=torch.cat(all_hidden,0)    
                        
                        all_label=torch.cat(all_label,1)    
                        all_label=torch.squeeze(all_label)
                        
                        
                        #score argumentation by cos similarity 
                        #all_score=score_argumentation(all_score, all_hidden, 8)
                        #_=score_argumentation_prompt(all_score, all_focus, 6)
                        
                        if len(all_score)!=len(all_source):
                            print(len(all_score), len(all_source)) 
                        
                        
                        select_number=int(torch.sum(all_label))
                        gold_top=torch.topk(all_label, select_number).indices
                        pred_top=torch.topk(all_score, self.config.ndoc).indices         
                        gold_top=gold_top.tolist()
                        pred_top=pred_top.tolist()
                        
                        source=[]
                        #for idx in gold_top:
                            #source.append(all_source[idx])
                        for idx in pred_top:
                            source.append(all_source[idx])   
                
                        source_score=[]
                        #for idx in gold_top:
                            #source_score.append(all_score[idx])
                        for idx in pred_top:
                            source_score.append(all_score[idx])     
                            
                        prompt=[]
                        #for idx in gold_top:
                            #prompt.append(all_focus[idx])
                        for idx in pred_top:
                            prompt.append(all_focus[idx])      
                            
                        hidden=[]
                        #for idx in gold_top:
                            #hidden.append((all_hidden[idx],idx))
                        for idx in pred_top:
                            hidden.append((all_hidden[idx],idx))
                            
                        '''
                        global_prompt=[]
                        global_sent=[]
                        global_score=[]
                        for idx,sent_and_idx in enumerate(hidden):
                            sent,indice=sent_and_idx
                            similar_sent=cos_retrieval(sent.unsqueeze(0),indice,all_hidden,3)
                            
                            global_prompt.append(all_focus[similar_sent[0]])
                            global_sent.append(all_source[similar_sent[0]])
                            global_score.append(all_score[similar_sent[0]])
                        '''
                        global_prompt=[]
                        global_sent=[]
                        global_score=[]
                        for idx,sent_and_idx in enumerate(hidden[:self.config.ndoc]):
                            sent,indice=sent_and_idx
                            
                            hidden_tensor=torch.cat([x[0].unsqueeze(0) for x in hidden],0)
                            similar_sent=cos_retrieval(sent.unsqueeze(0),idx,hidden_tensor,3)
                            
                            global_prompt.append(prompt[similar_sent[0]])
                            global_sent.append(source[similar_sent[0]])
                            global_score.append(source_score[similar_sent[0]])
                        
                        topk=self.config.ndoc
                        source=source[:topk]
                        source_score=source_score[:topk]   
                        prompt=prompt[:topk]
                        global_prompt=global_prompt[:topk]
                        global_sent=global_sent[:topk]                               
                        global_score=global_score[:topk]
                        
                        
                        if self.config.add_global_score == 1:
                            for i in range(len(source_score)):
                                source_score[i]+=global_score[i]
                        
                        source_score=torch.tensor([source_score]).cuda()
                        source_score=torch.softmax(source_score,1)
                        
                        #source=[source]
                        
                        summary=data_point[0][8][0]
                        query=data_point[0][7][0]
                        
                        with torch.no_grad():        
                            
                            prompt=[x[:self.config.local_window_number] for x in prompt]
                            global_prompt=[x[:self.config.global_window_number] for x in global_prompt]
                            
                            context_input_ids,context_attention_mask,labels=data_loader_from_ext_to_abs(source,summary,500,100,self.tokenizer_abs, self.config.pad_token_id, query,  prompt, global_prompt, global_sent, max_sent=self.config.ndoc,context=self.config.context)
                            outputs = self.generator.generate(context_input_ids=context_input_ids,
                                                               context_attention_mask=context_attention_mask,
                                                               doc_scores=source_score,
                                                               num_beams=1,
                                                               min_length=self.config.min_dec_steps,
                                                               max_length=self.config.max_dec_steps,
                                                               no_repeat_ngram_size=2,
                                                               length_penalty=1)
                
                
                            assert isinstance(outputs, torch.Tensor)
                            assert outputs.shape[0] == 1
                            decoded_pred = self.tokenizer_abs.batch_decode(outputs, skip_special_tokens=True)
                        pred=decoded_pred[0]    
                        
                        gold=summary
             
                         
                        scores = self.raw_rouge.get_scores(pred, gold)
                        r1.append(scores[0]['rouge-1']['f'])
                        r2.append(scores[0]['rouge-2']['f'])    

                        if data_loader_val.epoch == 2:
                            break

                        pred_list.append(pred)
                        gold_list.append(gold)  
                        source_list.append('xx')
                                                
                    
                    for sent in gold_list:
                        save_gold.write(sent.strip() + '\n')
                    for sent in pred_list:
                        save_pred.write(sent.strip() + '\n')
                    for sent in source_list:
                        save_source.write(sent.strip() + '\n')

        print(np.mean(r1),np.mean(r2),np.mean(r2_source),np.mean(length))



def argLoader():

    parser = argparse.ArgumentParser()
    
    
    #device
    
    parser.add_argument('--device', type=int, default=0)    
    
    parser.add_argument('--multi_gpu', type=int, default=0)        

    parser.add_argument('--multi_device', type=str, default='0,1,2,3')       
    # Do What
    
    parser.add_argument('--do_train', action='store_true', help="Whether to run training")

    parser.add_argument('--do_test', action='store_true', help="Whether to run test")
    
    parser.add_argument('--seed', type=int, default=10)

    parser.add_argument('--query', type=int, default=1)
    
    parser.add_argument('--local_window_number', type=int, default=3)
    
    parser.add_argument('--global_window_number', type=int, default=3)
    
    parser.add_argument('--local_window_size', type=int, default=8)  
    
    parser.add_argument('--add_global_score', type=int, default=0)      
    
    parser.add_argument('--ndoc', type=int, default=25)    
    
    parser.add_argument('--attention_rollout', type=int, default=1)
    
    parser.add_argument('--attention_rollout_layer', type=str, default='all')
    
    parser.add_argument('--context', type=str, default='c')

    
    
    parser.add_argument('--pretrained_model', type=str, default='roberta-base') 
    
    parser.add_argument('--pretrained_model_ext', type=str, default='roberta-base')    
    
    parser.add_argument('--pretrained_model_abs', type=str, default='facebook/bart-large')   
    
    parser.add_argument('--bos_token_id', type=int, default=0) 
    
    parser.add_argument('--pad_token_id', type=int, default=1) 
    
    parser.add_argument('--eos_token_id', type=int, default=2)
    #Preprocess Setting
    parser.add_argument('--max_summary', type=int, default=100)

    parser.add_argument('--max_article', type=int, default=350)    
    
    #Model Setting
    parser.add_argument('--hidden_dim', type=int, default=768)

    parser.add_argument('--emb_dim', type=int, default=768)
    
    parser.add_argument('--vocab_size', type=int, default=50264)      

    parser.add_argument('--lr', type=float, default=2e-6)     
    
    parser.add_argument('--eps', type=float, default=1e-10)
    
    parser.add_argument('--prefix_dropout', type=float, default=0)    
        
    parser.add_argument('--batch_size', type=int, default=1)  

    parser.add_argument('--true_batch_size', type=int, default=8)  

    parser.add_argument('--buffer_size', type=int, default=10)      
    
    parser.add_argument('--scale_embedding', type=int, default=0)  
    
    #lr setting

    parser.add_argument('--use_lr_decay', type=int, default=0)  
    
    parser.add_argument('--lr_decay_step', type=int, default=10000)  
    
    parser.add_argument('--lr_decay', type=float, default=1)  

    # Testing setting
    parser.add_argument('--beam_size', type=int, default=2)
    
    parser.add_argument('--max_dec_steps', type=int, default=100)
    
    parser.add_argument('--min_dec_steps', type=int, default=50)
    
    parser.add_argument('--test_model', type=str, default='')   
    
    parser.add_argument('--load_model', type=str, default='')  

    parser.add_argument('--ext_model', type=str, default='')  
    
    parser.add_argument('--save_path', type=str, default='')  
    
    parser.add_argument('--mid_start', type=int, default=1)
   
    # Checkpoint Setting
    parser.add_argument('--max_epoch', type=int, default=15)
    
    parser.add_argument('--train_set_len', type=int, default=60000)
    
    parser.add_argument('--savefreq', type=int, default=100)

    parser.add_argument('--checkfreq', type=int, default=1)    

    parser.add_argument('--startfreq', type=int, default=1)        
    
    args = parser.parse_args()
    
    return args





def main():
    args = argLoader()
    
    if args.multi_gpu==1:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.multi_device
    else:
        torch.cuda.set_device(args.device)

    print('CUDA', torch.cuda.current_device())
    
        
    if args.do_train == 1:
        x=Train(args)
        x.train_iter()
    if args.do_test==1:
        print('start testing the model')
        x = Test(args)
        x.test()


main()
        