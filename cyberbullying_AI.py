# 문장을 입력하면
# 악풀 검사 - 2가지 다른 분석기술을 이용하여 입력문장 검사
# 만약, 악풀이라면
# 적절한 문장 생성, 문장을 다른 비유적 표현으로 재진술하는 기능



# Check bad comment
import re
from happytransformer import HappyTextClassification
happy_tc = HappyTextClassification("BERT", "Hate-speech-CNERG/dehatebert-mono-english", 2)

# - 비유적 표현 생성 시작
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer_meta = AutoTokenizer.from_pretrained("figurative-nlp/t5-figurative-generation")
model_meta = AutoModelForSeq2SeqLM.from_pretrained("figurative-nlp/t5-figurative-generation")
# - 비유적 표현 생성 끝




# 다른 표현으로 재진술하기 - 시작
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# tokenizer_para = AutoTokenizer.from_pretrained("hetpandya/t5-small-tapaco")
# model_para = AutoModelForSeq2SeqLM.from_pretrained("hetpandya/t5-small-tapaco")


# create a function for the paraphrase
# def paraphrase(sentence):
   
#     sentence = "paraphrase: " + sentence + " </s>"
#     encoding = tokenizer_para.encode_plus(sentence,padding=True, return_tensors="pt")
#     input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

#     outputs = model_para.generate(
#     input_ids=input_ids, attention_mask=attention_masks,
#     do_sample=True, #샘플링 전략 사용
#     max_length=256, # 최대 디코딩 길이는 50
#     top_k=200, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
#     top_p=0.98, # 누적 확률이 95%인 후보집합에서만 생성
#     num_return_sequences=5) #3개의 결과를 디코딩해낸다

#     output = tokenizer_para.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)

#     return(output)


# 다른 표현으로 재진술하기 - 끝

# input Bad comment
#input_sent = "fucking ridiculous bullshit, fuck you"
input_sent = "fucking ridiculous bullshit"
#input_sent = "Great! I am very satisfied and I will cheer for you to continue to succeed."

result = happy_tc.classify_text(input_sent)
# print(result)
# print(result.label)
# print(result.score)

analysis_result = str(result.label)

# if analysis_result == "HATE":
#     print("Bullying words dectected!")

# 다른 모델을 이용해서 재검사
from transformers import AutoTokenizer, T5ForConditionalGeneration

ckpt = 'Narrativa/byt5-base-tweet-hate-detection'

tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = T5ForConditionalGeneration.from_pretrained(ckpt).to("cpu")

def classify_comment(tweet):

    inputs = tokenizer([tweet], padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    input_ids = inputs.input_ids.to('cpu')
    attention_mask = inputs.attention_mask.to('cpu')
    output = model.generate(input_ids, attention_mask=attention_mask)
    return tokenizer.decode(output[0], skip_special_tokens=True)

result2 = str(classify_comment(input_sent))
#print(result2)

# if result2 == "hate-speech":
#     print("Bullying words dectected!")

## 위의 2가지 모델을 가지고 더블체크 후 
# 두개의 분석 결과중 1개만 불싯언어로 판단되면  -> '당신의 마음이 얼마나 불편한지 공감이 간다. 하지만 다른 표현으로 말해주길 부탁할 수 있을까?"
# 두개의 분석결과가 모두 불싯언어로 나오면 -> AI가 당신의 언어를 감지했다. 여기에 당신의 말을 공지할 수 없어서 다른 말로 바꿔보고자 한다. 
# 비유적 표현으로 변환해보기 혹은 명언으로 변형해보기.
# 명언대체의 경우 --> 댓글 상황과 가장 잘 어울리는 명언으로 대체해보려고 한다.  --> 명언제시 하고 끝!


# 비유적 표현으로 변환기능 테스트
def metapherGen(input_text):
    input_text = "<m>" + input_text + "</m>"
    input_ids = tokenizer_meta(input_text, return_tensors="pt").input_ids  # Batch size 1
    outputs = model_meta.generate(input_ids)
    result__ = tokenizer_meta.decode(outputs[0], skip_special_tokens=True)
    #print("metapherGen result :", result__)
    return result__

# GPT-neo 를 이용한 문장 생성
from happytransformer import HappyGeneration
happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-125M")
from happytransformer import GENSettings

def Gpt_Gen():
    #args = GENSettings(no_repeat_ngram_size=2)
    top_k_sampling_settings = GENSettings(do_sample=True, early_stopping=False, top_k=50, temperature=0.7)
    Gen_re = happy_gen.generate_text("The way to change your uncomfortable mind in a good mood is ", args=top_k_sampling_settings)

    return Gen_re


import paraphrase as paraphrase

if analysis_result == "HATE" and result2 == "hate-speech":
    print("Malicious comments detected!")
elif analysis_result == "HATE" or result2 == "no-hate-speech":
    print("Comments found to make you feel a bit uncomfortable.")
    meta_re = metapherGen(input_sent)
    print("Replace with another expression 1 :", meta_re)
    meta_re_2 =Gpt_Gen()
    print("Replace with another expression 2 :", meta_re_2.text)
    paraph =  paraphrase.paraphrase_(input_sent)
    print("paraphrase expression 3 :", paraph)

elif analysis_result == "NON-HATE" or result2 == "hate-speech":
    print("Comments found to make you feel a bit uncomfortable.")
    meta_re = metapherGen(input_sent)
    print("Replace with another expression :", meta_re)
    meta_re_2 =Gpt_Gen()
    print("Replace with another expression 2 :", meta_re_2.text)
    paraph =  paraphrase.paraphrase_(input_sent)
    print("paraphrase expression 3 :", paraph)
else:
    print("No malicious comments found.")


