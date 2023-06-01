import nltk as nl

sentence="I am Sapthak"
# for the first usage nl.download('punkt')
tokens=nl.word_tokenize(sentence)
print(tokens)