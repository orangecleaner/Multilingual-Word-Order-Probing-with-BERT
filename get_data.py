import csv
import json
import random

target_langs = ["cmn", "jpn", "tur", "ara", "deu"]
input_file="sentences.csv"
output_file="word_order.json"
samples_per_lang = 500
lang_sentence={lang:[] for lang in target_langs}

WALS_WORD_ORDER = {
    "cmn": "SVO",
    'kor': 'SOV',
    "jpn": "SOV",
    "tur": "SOV",
    "ara": "VSO",
    "deu": "SVO",
}

with open(input_file,'r') as f:
    file=csv.reader(f,delimiter='\t')
    for row in file:
        id,lang,text=row[0],row[1],row[2]
        if lang in target_langs:
            lang_sentence[lang].append(text)

sample=[]
for lang in lang_sentence:
    random.shuffle(lang_sentence[lang])
    for sentence in lang_sentence[lang][:samples_per_lang]:
        sample.append({"lang":lang,
                       "sentence":sentence,
                       'label':WALS_WORD_ORDER[lang]})
    
with open(output_file, "w", encoding="utf-8") as f:
    for sample in sample:
        json.dump(sample, f, ensure_ascii=False)
        f.write("\n")

