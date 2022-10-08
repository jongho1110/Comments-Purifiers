#다른 표현으로 재진술하기 - 시작

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer_para = AutoTokenizer.from_pretrained("hetpandya/t5-small-tapaco")
model_para = AutoModelForSeq2SeqLM.from_pretrained("hetpandya/t5-small-tapaco")


def paraphrase_(sentence):

    sentence = "paraphrase: " + sentence + " </s>"
    encoding = tokenizer_para.encode_plus(sentence,padding=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

    outputs = model_para.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    do_sample=True, #샘플링 전략 사용
    max_length=256, # 최대 디코딩 길이는 50
    top_k=200, # 확률 순위가 50위 밖인 토큰은 샘플링에서 제외
    top_p=0.98, # 누적 확률이 95%인 후보집합에서만 생성
    num_return_sequences=5) #3개의 결과를 디코딩해낸다

    output = tokenizer_para.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)

    return(output)


