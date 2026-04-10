import streamlit as st
import numpy as np
import pickle 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
 #Load LSTM MODEl
 
model = load_model('Next_word_lstm.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#function to predict next word
def predict_next_word(model,tokenizer,text,max_sequence_len):
  token_list=tokenizer.texts_to_sequences([text])[0]#word to vector conversion
  if(len(token_list)>=max_sequence_len):
    token_list=token_list[-(max_sequence_len-1):]#Ensure sequence length matches max_sequence keeps last max_sequence_len-1 words to preserve context
  token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
  predicted=model.predict(token_list,verbose=0)
  predicted_word_index=np.argmax(predicted,axis=1)#show the word with max probability
  for word,index in tokenizer.word_index.items():
    if(index==predicted_word_index):
      return word
  return None

#streamlit app
st.title("Next Word Prediction with LSTM")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
  max_sequence_len=model.input_shape[1]+1
  next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
  st.write(f'Next word:{next_word}')

