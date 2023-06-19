from fastapi import FastAPI
import tensorflow as tf
import pickle
from tools import text_preprocessing , concatinate
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


app = FastAPI()
model = tf.keras.models.load_model('CNN_DOC_TYPE')
# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
max_len = 2000

LABELS = ['AAL','AssPur' ,'BizComb', 'CA', 'CP', 'Cncl', 'CoNo', 'ColAgy', 'ConsJ', 'Emp',
 'Ern', 'FOB', 'Gty', 'ICA', 'Ind', 'Indm' ,'MA', 'OCorp', 'OffCert', 'PR', 'Plg',
 'ProNo', 'PurSal', 'RO', 'RR', 'SA', 'SUB', 'ShrExc', 'Stlmt', 'StoPur', 'TSA',
 'TrSv', 'TrmSht', 'Undr', 'Wrnt']

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/document/")
async def say_hello(doc: str):
    text_tok =  text_preprocessing(doc, accented_chars=True, contractions=True,
                       convert_num=False, extra_whitespace=True,
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True,
                       stop_words=True)
    text_clean = concatinate(text_tok)
    print('text_clean',text_clean)
    sequences = tokenizer.texts_to_sequences([text_clean])

    # Padding
    X_tmp = pad_sequences(sequences, maxlen=max_len)
    predictions = model.predict(X_tmp)
    print('predictions',predictions)

    outputs = np.array(predictions) >= 0.207
    print('outputs',outputs)


    predicted_labels = [label for label, m in zip(LABELS , outputs[0]) if m]



    return {"doc_type": predicted_labels}
