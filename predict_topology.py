import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from captum.attr import IntegratedGradients
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import rcParams

fonts = {
    "zh": r"C:\Windows\Fonts\NotoSansSC-VF.ttf",   
    "ja": r"C:\Users\boyuntong\AppData\Local\Microsoft\Windows\Fonts\NotoSansJP-VariableFont_wght.ttf",  
    "ar": r"C:\Users\boyuntong\AppData\Local\Microsoft\Windows\Fonts\NotoSansArabic-VariableFont_wdth,wght.ttf",  
    "en": r"C:\Windows\Fonts\NotoSansSC-VF.ttf"  
}

def detect_lang(ch):
    code = ord(ch)
    if 0x4E00 <= code <= 0x9FFF:
        return "zh"  # 中文
    elif 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF:
        return "ja"  # 日文
    elif 0x0600 <= code <= 0x06FF:
        return "ar"  # 阿拉伯文
    else:
        return "en"  
    
def set_font(lang):
    """根据语言选择字体"""
    font_path = fonts.get(lang, fonts["en"])  # 默认英文
    return fm.FontProperties(fname=font_path)

model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = AutoModel.from_pretrained(model_name, output_hidden_states=True).eval()

class MLPProbe(nn.Module):
    def __init__(self,input_dim=768,hidden_dim=256,output_dim=3):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim)
        )
    
    def forward(self,x):
        return self.model(x)
    
probe=MLPProbe()
probe.load_state_dict(torch.load("mlp_probe_layer8.pt"))
probe.eval()

def forward_func(x):
    sentence_embedding = torch.mean(x, dim=0, keepdim=True)
    return probe(sentence_embedding)

sentence1 ='The dog chased the cat'
sentence2='犬が猫を追いかけた。'
sentence3='طارد الكلب القطة.'
sentences=[sentence1,sentence2,sentence3]
n=0
for sentence in sentences:
    inputs=tokenizer(sentence,return_tensors='pt')
    n+=1
    outputs=bert(**inputs)
    hidden_states=outputs.hidden_states
    layer_8=hidden_states[8][0]

    ig = IntegratedGradients(forward_func)

    baseline = torch.zeros_like(layer_8)
    pred = probe(torch.mean(layer_8, dim=0, keepdim=True)).argmax(dim=1).item()

    attributions, delta = ig.attribute(
        inputs=layer_8,
        baselines=baseline,
        target=pred,
        return_convergence_delta=True
    )

    token_importance=attributions.norm(p=2,dim=1)
    tokens=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    lang = detect_lang(sentence[0]) 
    prop = set_font(lang)

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(tokens)), token_importance.detach().numpy())
    plt.xticks(range(len(tokens)), tokens, rotation=45,fontproperties=prop)
    plt.title("Token Attribution for Word Order Prediction")
    plt.tight_layout()
    plt.savefig(f'{n}.png')
