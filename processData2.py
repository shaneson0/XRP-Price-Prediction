

import json


with open('./data/record_word.json', 'r') as fw:
    data = fw.readline()

Dic = {}
Data = json.loads(data, encoding='utf-8')
for key in Data.keys():
    value = Data[key]
    tempDic = {}
    for tuppleArray in value:
        tempDic[tuppleArray[0]] = int(tuppleArray[1])
    Dic[key] = tempDic


with open('./data/new_record_word.json', 'w', encoding='utf-8') as fw:
    json.dump(Dic, fw)








