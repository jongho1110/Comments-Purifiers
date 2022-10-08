import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from keytotext import pipeline

nlp = pipeline("k2t")

result = nlp(['When I see','I feel','because I need', 'Would you be willing to'])

print(result)