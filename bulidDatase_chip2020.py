from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer,AutoModel,BertForMaskedLM,AutoModelForMaskedLM
from tkitAutoTokenizerPosition import AutoTokenizerPosition,autoBIO,autoSpan
import json,os
import tkitJson
# from tqdm import tqdm
# dir(AutoTokenizerPosition)
from torch.utils.data import DataLoader, random_split,TensorDataset
import torch
from transformers import BertTokenizer, AlbertModel
tokenizer = BertTokenizer.from_pretrained("uer/chinese_roberta_L-8_H-256")
from tqdm.auto import tqdm
Data="data/val_data.json"


MaxLen=128
# 创建输出文件夹
if os.path.isdir("data"):
    pass
else:
    os.mkdir("data")


datas=[]
for file in [Data]:
    Tjson=tkitJson.Json(file)
    APo=AutoTokenizerPosition(tokenizer)
    for i,it in tqdm(enumerate(Tjson.load())):
        # print(it)
    #     info.find('a')
        data=[]
        for one in it['spo_list']:
            word=one['subject']
            wType=one['subject_type']
            for s_start,s_end in APo.fixPosition(it['text'],word):
#             print(s_start,s_end)
                WordList=APo.getWordList(it['text'])
                data.append({"start":s_start,"end":s_end,"type":wType})  

            word=one['object']["@value"]
            wType=one['object_type']["@value"]
            for s_start,s_end in APo.fixPosition(it['text'],word):
#             print(s_start,s_end)
                WordList=APo.getWordList(it['text'])
                data.append({"start":s_start,"end":s_end,"type":wType})  
    
            word=one['predicate']
            wType="关系"
            for s_start,s_end in APo.fixPosition(it['text'],word):
                WordList=APo.getWordList(it['text'])
                data.append({"start":s_start,"end":s_end,"type":wType})  
#             break
#         print(data)
        datas.append({"text":it['text'],"wordList":WordList,"tag":data,"data":it})
#         if i>10:
# #             print
#             break


labels={"O":0}
texts=[]
tags=[]
# 构建bio格式数据集
bio=autoBIO()
for it in tqdm(datas):
    # print(it)
    it=bio.bulid(it)
    # print(it)
    tag=["O"]+it['tagList']+["O"]*MaxLen
    tag=tag[:MaxLen]
    tags.append(tag)
    texts.append(it["text"])
    
    for t in tag:
        labels[t]=0

# 构建标签列表


labelsList=list(labels.keys())
print("labelsList",labelsList)

print("len labelsList",len(labelsList))

with open(os.path.join("data","labels.json"),'w') as f:
    json.dump(labelsList,f)
# print(datas[:1])



def tags2id(tags,labels):
    for i,it in  enumerate (tags):
        tags[i] = labels.index(it)
    return tags
    pass


for i,tag in enumerate( tags):
    tags[i]=tags2id(tag,labelsList)
    
    













# print(tags)
textTensor=tokenizer(texts, padding="max_length",max_length=MaxLen,  truncation=True,return_tensors="pt")
tagsTensor=torch.Tensor(tags)
myDataset=TensorDataset(textTensor["input_ids"],textTensor["token_type_ids"],textTensor["attention_mask"],tagsTensor)
# print(textTensor)

# torch.save(myDataset,"out/dataset.bin")


fl=len(myDataset)
trainl=int(fl*0.7)
testl=int(fl*0.15)
vall=fl-trainl-testl
train_set,val_set,test_set=random_split(myDataset, [trainl,vall,testl])

torch.save(train_set,"data/train.pkt")
torch.save(val_set,"data/val.pkt")
torch.save(test_set,"data/test.pkt")