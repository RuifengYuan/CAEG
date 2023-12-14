from call_rouge import test_rouge, rouge_results_to_str
import glob


model_list=glob.glob('result/*')
try:
    model_list.remove('result/rouge')
except:
    pass

for i in model_list:
    next_path= i+'/*'

    next_path= i+'/*'
    
    raw_chech_point_list=glob.glob(next_path)
    if '_cand.txt' in raw_chech_point_list[0]:
        test_model=raw_chech_point_list[0].split('_cand.txt')[0]
    if '_gold.txt' in raw_chech_point_list[0]:
        test_model=raw_chech_point_list[0].split('_gold.txt')[0]
    
    can_path = test_model+'_cand.txt'
    
    gold_path = test_model+'_gold.txt'
    
    rouges = test_rouge('result/rouge', can_path, gold_path)
    print(test_model)
    print(rouge_results_to_str(rouges))
    print('______________________________________')

