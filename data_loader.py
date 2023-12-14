import glob
import random
import struct
import json
import re
import torch
import csv
import argparse
from transformers import BartTokenizer
from rouge import Rouge
import nltk
from nltk import word_tokenize




REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",

         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"', "\n": '',":":',',"\'":"'",'<s>':'','</s>':''}

def clean(x):
    return re.sub(r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''|\n|:", lambda m: REMAP.get(m.group()), x)

def clean_str(s):
    forbidden=['b"','-lrb-','-rrb-','-','“','"',"'","`",'``',"''","b'",'/','\\','\\n','-','<s>','</s>']
    for i in forbidden:
        s=s.replace(i,'')
    return s

def end_replace(s):
    forbidden=['!', '?',';']

    for i in forbidden:
        s=s.replace(i,'.')
   
    return s






class data_loader():

    def __init__(self, part, config, tokenizer, sum_type, load_qmsum=1, load_arxiv=0):

        super(data_loader,self).__init__()  
      
        self.part=part
        self.tokenizer=tokenizer
        self.config=config
        self.raw_rouge=Rouge()
        
        random.seed(self.config.seed)
        
        self.count=0
        self.epoch=0

        self.max_epoch=config.max_epoch
        self.buffer_size=config.buffer_size
        self.batch_size=config.batch_size
        self.true_batch_size=config.true_batch_size
        self.max_article=config.max_article
        self.max_summary=config.max_summary

        '''
        self.max_epoch=1
        self.buffer_size=128
        self.batch_size=4
        self.max_article=512
        self.max_summary=100
        '''
        self.load_qmsum=load_qmsum
        self.load_querysum=load_querysum
        self.load_arxiv=load_arxiv
        self.sum_type=sum_type
    
        if load_qmsum == 1:            
            if self.part == 'train':
                self.qmsum_source = 'data/qmsum_ext_new/train.txt'
            if self.part == 'val':
                self.qmsum_source = 'data/qmsum_ext_new/val.txt'
            if self.part == 'test':
                self.qmsum_source = 'data/qmsum_ext_new/test.txt'

        if load_arxiv == 1:            
            if self.part == 'train':
                self.arxiv_source = 'data/arxiv/train.txt'
            if self.part == 'val':
                self.arxiv_source = 'data/arxiv/val.txt'
            if self.part == 'test':
                self.arxiv_source = 'data/arxiv/test.txt'        


        if load_qmsum == 1:
            self.data_generator=self.next_data_qmsum()
  
        if load_arxiv == 1:
            self.data_generator=self.next_data_arxiv() 
        
        
        if self.sum_type == 'extractive':
            self.batch_generator=self.next_batch_ext()
        if self.sum_type == 'abstractive':
            self.batch_generator=self.next_data_point()        
        if self.sum_type == 'inference':
            self.batch_generator=self.next_data_point()            

    
    def next_data_arxiv(self):
        buffer=[]
        for epoch in range(self.max_epoch):
            self.epoch=self.epoch+1
            
            with open(self.arxiv_source,'r') as load_f:
                data_list=load_f.readlines()
                
            data=[]
            for i in data_list:
                load_dict = eval(i)
                data.append(load_dict)

            if self.part == 'train':
                random.shuffle(data)
            else:
                pass
                        
            
            for data_point in data:
                

                query=''
                answer=data_point['summary'] 
                source=data_point['article']          
                extract_idx=data_point['oracle_id']
                
                
                extract_answer=[source[ora] for ora in extract_idx]

                answer=self.tokenize(answer)     
                source=[self.tokenize(s) for s in source]
                extract_answer=[self.tokenize(s) for s in extract_answer]
                
                summary=answer.split('. ')
                source_splited_token=[]
                source_splited=[]
                one_span_token=[]
                one_span=[]
                count=0
                for sent in source:
                    sent_token=self.tokenizer.encode(sent)
                    if (count+len(sent_token)) > 500:
                        source_splited_token.append(one_span_token)
                        source_splited.append(one_span)
                        
                        count=len(sent_token)
                        one_span_token=[sent_token]
                        one_span=[sent]                        
                    else:
                        one_span_token.append(sent_token)
                        one_span.append(sent)
                        count+=len(sent_token)
                        
                if len(one_span_token) != 0:
                    source_splited_token.append(one_span_token)
                    source_splited.append(one_span)
                     
                all_label=[]
                count_pos=0
                count_neg=0
                for span in source_splited:
                    label=[]
                    for sent in span:
                       if sent in extract_answer:
                           label.append(float(1))
                           count_pos+=1
                       else:
                           label.append(float(0))
                           count_neg+=1
                    all_label.append(label)

                label_weight=[count_neg/count_pos]*len(all_label)
                
                query=[query]*len(all_label)     

                buffer.append((source_splited_token,source_splited,query,summary,all_label,label_weight))     
                        
                if len(buffer) == self.buffer_size:
                    yield buffer
                    buffer=[]
                        
        print ("data_generator completed reading all datafiles for all epoches. No more data.")                       
        return 0


    
   
    def next_data_qmsum(self):
        buffer=[]
        for epoch in range(self.max_epoch):
            self.epoch=self.epoch+1
            data_path = open(self.qmsum_source, 'r')   
            
            data=data_path.readlines()   
                
            if self.part == 'train':
                random.shuffle(data)
            else:
                pass
                        
            for i in data:
                
                data_point=eval(i)
                
                query=data_point['query']
                answer=data_point['answer']   
                source=data_point['source']            
                extract_answer=data_point['extract_answer'] 
                
                query=self.tokenize(query)
                answer=self.tokenize(answer)     
                source=[self.tokenize(s) for s in source]
                extract_answer=[self.tokenize(s) for s in extract_answer]   
                
                
                summary=answer.split('. ')
                source_splited_token=[]
                source_splited=[]
                one_span_token=[]
                one_span=[]
                count=0
                for sent in source:
                    sent_token=self.tokenizer.encode(sent)
                    if (count+len(sent_token)) > 500:
                        source_splited_token.append(one_span_token)
                        source_splited.append(one_span)
                        
                        count=len(sent_token)
                        one_span_token=[sent_token]
                        one_span=[sent]                        
                    else:
                        one_span_token.append(sent_token)
                        one_span.append(sent)
                        count+=len(sent_token)
                        
                if len(one_span_token) != 0:
                    source_splited_token.append(one_span_token)
                    source_splited.append(one_span)
                     
                all_label=[]
                count_pos=0
                count_neg=0
                for span in source_splited:
                    label=[]
                    for sent in span:
                       if sent in extract_answer:
                           label.append(float(1))
                           count_pos+=1
                       else:
                           label.append(float(0))
                           count_neg+=1
                    all_label.append(label)

                label_weight=[count_neg/count_pos]*len(all_label)
                
                query=[query]*len(all_label)     

                buffer.append((source_splited_token,source_splited,query,summary,all_label,label_weight))     
                        
                if len(buffer) == self.buffer_size:
                    yield buffer
                    buffer=[]
                        
        print ("data_generator completed reading all datafiles for all epoches. No more data.")                       
        return 0
    
    
 
    
    def next_batch_ext(self):
        while(True):
            
            count=0
            batch_source_id=[]
            batch_clss=[]
            batch_source=[]
            batch_query=[]
            batch_label=[]
            batch_weight=[]
            
            data = self.data_generator.__next__()
            for source in data:
                source_splited_token,source_splited,query,summary,all_label,label_weight=source                
                for idx,span in enumerate(source_splited_token):
                    if count == self.batch_size:
                        
                        batch_source_id,batch_source_id_mask=self.pad_with_mask(batch_source_id, pad_id=self.config.pad_token_id)
                        batch_source_id=torch.tensor(batch_source_id)
                        batch_source_id_mask=torch.tensor(batch_source_id_mask) 
                        
                        batch_clss,batch_clss_mask=self.pad_with_mask(batch_clss, pad_id=self.config.pad_token_id)
                        batch_clss=torch.tensor(batch_clss)
                        batch_clss_mask=torch.tensor(batch_clss_mask)  
                        
                        batch_label,batch_label_mask=self.pad_with_mask(batch_label, pad_id=self.config.pad_token_id)
                        batch_label=torch.tensor(batch_label)
                        batch_label_mask=torch.tensor(batch_label_mask)                          
                        
                        batch_source_id=batch_source_id.cuda()
                        batch_source_id_mask=batch_source_id_mask.cuda()
                        batch_clss=batch_clss.cuda()
                        batch_clss_mask=batch_clss_mask.cuda()
                        batch_label=batch_label.cuda()
                        batch_label_mask=batch_label_mask.cuda()

                        
                        yield [batch_source_id,
                               batch_source_id_mask,
                               batch_clss,
                               batch_clss_mask,
                               batch_label,
                               batch_weight,
                               batch_source,
                               batch_query]                        

                        count=0
                        batch_source_id=[]
                        batch_clss=[]
                        batch_source=[]
                        batch_query=[]
                        batch_label=[]
                        batch_weight=[]          
                        
                    else:
                        
                        if 1 not in all_label[idx]:
                            continue
                        
                        cls_token=self.config.eos_token_id
                        
                        one_source_id=self.tokenizer.encode(query[idx])
                        for sent in source_splited_token[idx]:
                            one_source_id+=sent                        
                            
                        one_source_id=one_source_id[:500]
                        
                        one_clss=[]
                        for t_dix,token in enumerate(one_source_id):
                            if token == cls_token:
                                one_clss.append(t_dix)
                        one_clss=one_clss[1:]
                        
                        one_label=all_label[idx][:len(one_clss)]

                        
                        assert  len(one_clss) == len(one_label),'sentence number is not correct between label and input'

                        batch_source_id.append(one_source_id)
                        batch_clss.append(one_clss)                        
                        batch_source.append(source_splited[idx])[:len(one_clss)]
                        batch_query.append(query[idx])
                        batch_label.append(one_label)
                        #batch_weight.append(label_weight[idx])          
                        batch_weight.append(len(one_label)/sum(one_label))
                        #batch_weight.append(0)
                        count+=1
            
            
    def next_data_point(self):
        while(True):
            

            
            data = self.data_generator.__next__()
            for source in data:
                source_splited_token,source_splited,query,summary,all_label,label_weight=source    
                full_data_point=[]
                #print(len(source_splited_token))
                #print(len(source_splited_token))
                #print(query[0])
                for idx,span in enumerate(source_splited_token[:50]):

                    count=0
                    batch_source_id=[]
                    batch_clss=[]
                    batch_source=[]
                    batch_query=[]
                    batch_label=[]
                    batch_weight=[]          
                    batch_summary=[]
                    
                    cls_token=self.config.eos_token_id
                    #one_clss=[]
                    q=query[idx]
                    if '.' not in q:
                        q+=' . '
                    one_source_id=self.tokenizer.encode(q)
                    for sent in source_splited_token[idx]:
                        one_source_id+=sent
                        #one_source_id+=[cls_token]
                        #one_clss.append(len(one_source_id)-1)
                    one_source_id=one_source_id[:510]
                    #one_clss=[n for n in one_clss if n < 512]
                    if one_source_id[-1] != cls_token:                        
                        one_source_id.append(cls_token)
                    
                    one_clss=[]
                    for t_dix,token in enumerate(one_source_id):
                        if token == cls_token:
                            one_clss.append(t_dix)                                           
                    
                                                    
                    one_clss=one_clss[1:]
                    
                    one_label=all_label[idx][:len(one_clss)]

                    
                    assert  len(one_clss) == len(one_label),'sentence number is not correct between label and input'

                    batch_source_id.append(one_source_id)
                    batch_clss.append(one_clss)                        
                    batch_source.append(source_splited[idx][:len(one_clss)])
                    batch_query.append(query[idx])
                    batch_label.append(one_label)
                    
                    if self.sum_type == 'abstractive':
                        batch_summary.append(' . '.join(summary))                     
       
                    batch_weight.append(1)
                    count+=1            
        
        
                    batch_source_id,batch_source_id_mask=self.pad_with_mask(batch_source_id, pad_id=self.config.pad_token_id)
                    batch_source_id=torch.tensor(batch_source_id)
                    batch_source_id_mask=torch.tensor(batch_source_id_mask) 
                    
                    batch_clss,batch_clss_mask=self.pad_with_mask(batch_clss, pad_id=self.config.pad_token_id)
                    batch_clss=torch.tensor(batch_clss)
                    batch_clss_mask=torch.tensor(batch_clss_mask)  
                    
                    batch_label,batch_label_mask=self.pad_with_mask(batch_label, pad_id=self.config.pad_token_id)
                    batch_label=torch.tensor(batch_label)
                    batch_label_mask=torch.tensor(batch_label_mask)                          
                    
                    batch_source_id=batch_source_id.cuda()
                    batch_source_id_mask=batch_source_id_mask.cuda()
                    batch_clss=batch_clss.cuda()
                    batch_clss_mask=batch_clss_mask.cuda()
                    batch_label=batch_label.cuda()
                    batch_label_mask=batch_label_mask.cuda()
                    '''
                    print(batch_source_id)
                    print(batch_clss)
                    print(batch_source)
                    
                    xxx=1
                    yyy=0
                    assert xxx==yyy
                    '''
                    if self.sum_type == 'abstractive':
                        full_data_point.append((batch_source_id,
                                               batch_source_id_mask,
                                               batch_clss,
                                               batch_clss_mask,
                                               batch_label,
                                               batch_weight,
                                               batch_source,
                                               batch_query,
                                               batch_summary))
                    else:
                        full_data_point.append((batch_source_id,
                                               batch_source_id_mask,
                                               batch_clss,
                                               batch_clss_mask,
                                               batch_label,
                                               batch_weight,
                                               batch_source,
                                               batch_query))       
                    
                yield full_data_point


    def load_data(self):
        self.count=self.count+1
        return self.batch_generator.__next__()
                    
            
    def get_sort(self, x):
        return len(x[0])


    def pad_with_mask(self, data, pad_id=0, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
            
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        
        pad_mask = [[1] * len(d) + [0] * (width - len(d)) for d in data]
        return rtn_data,pad_mask 




    def tokenize(self,sent):
        tokens = ' '.join(word_tokenize(sent.lower()))
        return tokens


    def greedy_selection(self, doc_sent_list, abstract_sent_list, summary_size):
        selected = []
        max_rouge = 0.0
        reference=''
        doc_sent_list=doc_sent_list[:15]
        for i in abstract_sent_list:
            reference+=i
            reference+=' . '
        for s in range(summary_size):
            cur_max_rouge = max_rouge
            cur_id = -1
            for i in range(len(doc_sent_list)):
                if (i in selected):
                    continue
                c = selected + [i]
                candidates = ''
                for j in c:
                    candidates+=doc_sent_list[j]
                    candidates+=' . '
                scores = self.raw_rouge.get_scores(hyps=candidates, refs=reference)
                rouge_score = (scores[0]['rouge-1']['f']+scores[0]['rouge-2']['f'])/2
                if rouge_score > cur_max_rouge:
                    cur_max_rouge = rouge_score
                    cur_id = i
            if (cur_id == -1):
                break
            selected.append(cur_id)
            max_rouge = cur_max_rouge
            
        select_sent=[]
        for i in sorted(selected):
            select_sent.append(doc_sent_list[i])
        return select_sent    
