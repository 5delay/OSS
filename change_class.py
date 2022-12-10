import os
from tqdm import tqdm

data_dir = './Data/DocLayNet/labels/'
save_dir = './Data/4class_DocLayNet/labels/'

for file_name in tqdm(os.listdir(data_dir)):
    f = open(data_dir + file_name, 'r')    
    
    while True:
    
        line = f.readline()
        if not line: break
    
        if line[0]=='2':
            new_f = open(save_dir + file_name, 'a')
            anno = '0'+line[1:]
            new_f.write(anno)
            new_f.close()
    
        elif line[0] == '6':
            new_f = open(save_dir + file_name, 'a')
            anno = '1'+line[1:]
            new_f.write(anno)
            new_f.close()
    
        elif line[0] == '8':
            new_f = open(save_dir + file_name, 'a')
            anno = '2' + line[1:]
            new_f.write(anno)     
            new_f.close()
        
        elif line[0] == '0' or line[0] == '3' or line[0] == '9' or line[0] == '7' or line[0] == '10':
            new_f = open(save_dir + file_name, 'a')
            anno = '3' + line[1:]
            new_f.write(anno)     
            new_f.close()
            
        else : continue  
    
    f.close()

