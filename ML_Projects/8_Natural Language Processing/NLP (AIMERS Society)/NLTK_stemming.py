import nltk
from nltk import PorterStemmer
#nltk.download('popular') downloads popular words
stemmer=PorterStemmer()

sentence="playing gaming running played writing dubbing"
words=nltk.word_tokenize(sentence)

for word in words:
    stemmed_word=stemmer.stem(word)
    print(f"{word}, ->, {stemmed_word}")
    #or this can be used
    #print(word, "->", stemmed_word)


