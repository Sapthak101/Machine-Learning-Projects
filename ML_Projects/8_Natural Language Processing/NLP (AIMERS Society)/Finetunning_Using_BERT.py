from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
# corpus means large texts
#it is a light weight model for question and answering using the Google BERT
# Load pre-trained tokenizer and model for question-answering
# the pre-trained tokenizer from autotokenizer is used to tokenize the context and prompt in an understandable manner for the model to analze it
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
#here we are using google bert, used for analyzing the text (question-answer)
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

# Define context and question
context = "Sapthak is from Bangladesh, Sapthak's age is 20, Mohandas Karamchand Gandhi was an Indian independence activist who was also the leader of the Indian National Congress. He is also known as Mahatma Gandhi or 'bapu'.SaiSatish is Gandhi's best friend"
context1 = "Life is game, Just enjoy each and every movement"

question = "What is Mahatma Gandhi known for?"
question1 = "What is other name for Mahatma Gandhi"
question2 = "Who is SaiSatish"
question3 = "What is the age of Sapthak"

# Use pipeline to generate answer
#ultimate model with the combination of the base model and the tokenizer
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
result = nlp(question=question2, context=context)
model.save_pretrained('/content/sample_data/1')
# Print answer
#here score in the result means confidence
#the result is in a dictionary format consisting of vectors values
print(result['answer'])