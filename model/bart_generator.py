# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import sys
sys.path.insert(0, os.getcwd())     
sys.path.insert(0, 'G:\project\fusion_bart\model') 
sys.path.insert(0, '/home/ruifeng/project/fusion_bart/model') 
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartConfig

class bart_generator(nn.Module):

    def __init__(self, config):

        super(bart_generator,self).__init__()  
      
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        self.config=config
        
       
    def forward(self, input_ids, input_mask, summary_ids, summary_mask):
        
        #input_id: (batch_size, sequence_length)
        
        outputs = self.model(input_ids=input_ids,attention_mask=input_mask, decoder_input_ids = summary_ids)
        
        return outputs 

    
    def inference(self, input_ids, input_mask, use_beam = 1):
        
        #input_id: (batch_size, sequence_length)
        #compressed_emb: (batch_size,1,hidden_size)
        
        if use_beam == 0:
            outputs = self.generate_without_beam_search(input_ids, input_mask)
        else:
            outputs = self.beam_search(input_ids, input_mask)          
            
        return outputs
    
    
    
    def generate_without_beam_search(self, input_ids,input_mask, max_step=80, min_step=30):
        
        generate_seq=[2]
        
        prepeared_decode_ids=torch.tensor([generate_seq]).cuda()
        
        outputs = self.model(input_ids=input_ids,attention_mask=input_mask, decoder_input_ids = prepeared_decode_ids, use_cache = 1)        
        
        encode_out = outputs.encoder_last_hidden_state 

        memory = outputs.past_key_values 
        
        output_seq=outputs.logits 
        
        output_seq = torch.softmax(output_seq, 2)      
        
        current_token = int(output_seq[0][-1].topk(1).indices)
        
        generate_seq.append(current_token)
        
        for i in range(max_step):
            
            prepeared_decode_ids=torch.tensor([[generate_seq[-1]]]).cuda()
            
            outputs = self.model(encoder_outputs=encode_out, attention_mask=input_mask, decoder_input_ids = prepeared_decode_ids,past_key_values = memory, use_cache = 1)     
            
            memory = outputs.past_key_values 
            
            output_seq=outputs.logits 
            
            output_seq = torch.softmax(output_seq, 2)      
            
            current_token = int(output_seq[0][-1].topk(1).indices)
            
            generate_seq.append(current_token)          
            
            if current_token == 2 and i >= min_step:
                
                break
            
        return generate_seq
    
    
    def beam_search(self,input_ids,input_mask, max_step=120, beam_size=4, min_step=60):
        
        generate_seq=[2]
        
        prepeared_decode_ids=torch.tensor([generate_seq]).cuda()
        
        outputs = self.model(input_ids=input_ids,
                             attention_mask=input_mask,
                             decoder_input_ids = prepeared_decode_ids, 
                             use_cache = 1)        
        
        encode_out = outputs.encoder_last_hidden_state 

        memory = outputs.past_key_values 
        
        output_seq=outputs.logits 
        
        output_seq = torch.softmax(output_seq, 2)      
        
        current_token = int(output_seq[0][-1].topk(1).indices)
        
        generate_seq.append(current_token)        
        

        beams = [Beam(tokens=generate_seq,    
                      log_probs=[0,0],    
                      state=memory)
                 for _ in range(beam_size)]        
        
        encode_out = torch.cat([encode_out for i in range(beam_size)],0).cuda()
        input_mask = torch.cat([input_mask for i in range(beam_size)],0).cuda()

        
        results = []
        steps = 0
        while steps < max_step and len(results) < beam_size:  
            
            all_state=[]
            for layer in range(len(beams[0].state)):
                one_layer_tensor=[]
                for t in range(4):
                    all_tensor=[]
                    for h in beams:      
                        all_tensor.append(h.state[layer][t])    
                    stacked_tensor = torch.cat(all_tensor,0).cuda()
                    one_layer_tensor.append(stacked_tensor)
                
                all_state.append(tuple(one_layer_tensor))
            all_state=tuple(all_state)
            
            stacked_input_id = torch.tensor([[h.latest_token] for h in beams])
            stacked_input_id = stacked_input_id.cuda()
            
            outputs = self.model(encoder_outputs=encode_out, 
                                 attention_mask=input_mask,
                                 decoder_input_ids = stacked_input_id, 
                                 past_key_values = all_state, 
                                 use_cache = 1)                   
            
            memory = outputs.past_key_values 
            output_seq=outputs.logits
            output_seq=output_seq[:,-1:,:]
            output_seq = torch.softmax(output_seq, 2)             
            output_seq = torch.log(output_seq)

            topk_log_probs, topk_ids = torch.topk(output_seq, beam_size * 2)            

            all_beams = []
            num_orig_beams = 1 if steps == 0 else beam_size
            for i in range(num_orig_beams):
                memory_i=[]
                for layer in range(len(beams[0].state)):
                    one_layer_tensor_i=[]
                    for t in range(4):   
                        one_tensor=memory[layer][t][i]
                        one_tensor=torch.unsqueeze(one_tensor,0)
                        one_layer_tensor_i.append(one_tensor)
                    memory_i.append(tuple(one_layer_tensor_i))
                memory_i=tuple(memory_i)
                
                h=beams[i]
                for j in range(beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, 0, j].item(),
                                   log_prob=topk_log_probs[i, 0, j].item(),
                                   state=memory_i)

                    all_beams.append(new_beam)


            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == 2:
                    if steps >= min_step:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == beam_size or len(results) == beam_size:
                    break
            steps=steps+1       
     
        
        if len(results) == 0:    
            results = beams        
        beams_sorted = self.sort_beams(results)    
        return beams_sorted[0].tokens   
    
    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True) 
    
    
    
    
class Beam(object):

  def __init__(self, tokens, log_probs, state):

    self.tokens = tokens

    self.log_probs = log_probs

    self.state = state




  def extend(self, token, log_prob, state):

    return Beam(tokens = self.tokens + [token],

                      log_probs = self.log_probs + [log_prob],

                      state = state)

  @property

  def latest_token(self):

    return self.tokens[-1]



  @property

  def avg_log_prob(self):

    return sum(self.log_probs) / len(self.log_probs)

    
