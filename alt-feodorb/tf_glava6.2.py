# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 20:40:06 2019

@author: feodorb
"""
import numpy as np
import os
   
input_fname = "file.txt"
output_fname = "outp.txt"
model_fname = "model_lstm"

START_CHAR = '\b'
END_CHAR = '\t'
PADDING_CHAR = '\a'

chars = set([START_CHAR, '\n', END_CHAR])

with open(input_fname, encoding='utf8') as f:
    for line in f:
        chars.update(list(line.strip().lower()))
        
char_indices = {c: i for i,c in enumerate(sorted(list(chars)))}
char_indices[PADDING_CHAR] = 0
indices_to_chars = {i : c for c, i in char_indices.items()}
num_chars = len(chars)

def get_one(i, sz):
    res = np.zeros(sz)
    res[i] = 1
    return res

char_vectors = {
        c: (np.zeros(num_chars) if c == PADDING_CHAR else get_one(v, num_chars))
        for c, v in char_indices.items()
        }

sentence_end_markers = set('.!?')

sentences = []
current_sentence = ''
with open(input_fname, 'r', encoding='utf8') as f:
    for line in f:
        s = line.strip().lower()
        if len(s) > 0:
            current_sentence += s + '\n'
        if (len(s) == 0 or s[-1] in sentence_end_markers):
            current_sentence = current_sentence.strip()
            if (len(current_sentence)) > 10:
                sentences.append(current_sentence)
            current_sentence = ''
            
def get_matrices(sent):
    max_sentence_len= np.max([len(x) for x in sent])
    X = np.zeros((len(sent), max_sentence_len, len(chars)), dtype=np.bool)
    y = np.zeros((len(sent), max_sentence_len, len(chars)), dtype=np.bool)
    
    for i, sentence in enumerate(sent):
        char_seq = (START_CHAR + sentence + END_CHAR).ljust(max_sentence_len+1, PADDING_CHAR)
        
        for t in range(max_sentence_len):
            X[i, t, :] = char_vectors[char_seq[t]]
            y[i, t, :] = char_vectors[char_seq[t+1]]
    return X, y

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM, TimeDistributed, Activation, Concatenate, Input

vec = Input(shape=(None, num_chars))
l1 = CuDNNLSTM(128, return_sequences=True)(vec)
l1_d=Dropout(0.2)(l1)

input2 = Concatenate()([vec, l1_d])
l2 = CuDNNLSTM(128, return_sequences=True)(input2)
l2_d=Dropout(0.2)(l2)

input3 = Concatenate()([vec, l2_d])
l3 = CuDNNLSTM(128, return_sequences=True)(input3)
l3_d=Dropout(0.2)(l3)

inputd = Concatenate()([l1_d, l2_d, l3_d])
dense3 = TimeDistributed(Dense(num_chars))(inputd)
output_res = Activation('softmax')(dense3)

model = Model(inputs=vec, outputs=output_res)

from tensorflow.keras.optimizers import Adam
model.compile(loss='categorical_crossentropy', optimizer=Adam(clipnorm=1.), metrics=['accuracy'])

test_indices = np.random.choice(range(len(sentences)), int(len(sentences) * 0.05))
sentences_train = [sentences[x] for x in set(range(len(sentences))) - set(test_indices)]
sentences_test = [sentences[x] for x in test_indices]
sentences_train = sorted(sentences_train, key = lambda x : len(x))
X_test, y_test = get_matrices(sentences_test)
batch_size=64
def generate_batch():
    while True:
        for i in range(int(len(sentences_train) / batch_size)):
            sentences_batch = sentences_train[i * batch_size: (i+1) * batch_size]
            yield get_matrices(sentences_batch)
            
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

import tensorflow as tf

class CharSampler(tf.keras.callbacks.Callback):
        def __init__(self, char_vectors, model):
            self.char_vectors = char_vectors
            self.model = model
        
        def on_train_begin(self, logs=None):
            print("\nTraining began")
            if os.path.isfile(output_fname):
                os.remove(output_fname)
        
        def sample( self, preds, temperature=1.0):
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            probas = np.random.multinomial(1, preds, 1)
            return np.argmax(probas)
         
        def sample_one(self, T):
            result = START_CHAR
            while len(result)<500:
                Xsampled = np.zeros( (1, len(result), num_chars) )
                for t,c in enumerate(list(result)):
                    Xsampled[0,t,:] = self.char_vectors[ c ]
                ysampled = self.model.predict( Xsampled, batch_size=1 )[0,:]
                yv = ysampled[len(result)-1,:]
                selected_char = indices_to_chars[ self.sample( yv, T ) ]
                if selected_char==END_CHAR: 
                    break
                 
                result = result + selected_char
            return result
         
        def on_epoch_end(self, epoch, logs=None):
            print("\nEpoch ended  %d" % epoch)
            if epoch % 50 == 0:
                print("\nEpoch %d text sampling:" % epoch)
                with open( output_fname, 'a', encoding='utf8' ) as outf:
                    outf.write( '\n===== Epoch %d =====\n' % epoch )
                    for T in [0.3, 0.5, 0.7, 0.9, 1.1]:
                        print('\tsampling, T = %.1f...' % T)
                        for _ in range(5):
                            self.model.reset_states()
                            res = self.sample_one(T)
                            outf.write( '\nT = %.1f\n%s\n' % (T, res[1:]) )   

    
cb_sampler = CharSampler(char_vectors, model)
cb_checkpoint = ModelCheckpoint('checkpoint.txt')
cb_logger = CSVLogger('sin_l/' + model_fname + '.log')


#X_tr, y_tr = get_matrices(sentences_train)
#model.fit(x=X_tr, y=y_tr, batch_size=batch_size,
#                    epochs=1000, verbose=True, validation_data = (X_test, y_test),
#                    callbacks=[cb_logger, cb_sampler, cb_checkpoint])


model.fit_generator(generate_batch(), int(len(sentences_train) / batch_size) * batch_size,
                    epochs=1000, verbose=True, validation_data = (X_test, y_test),
                    callbacks=[cb_logger, cb_sampler, cb_checkpoint])





            

