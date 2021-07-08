# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:03:41 2021

@author: rubab
"""

import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
from flask import Flask, request, jsonify, render_template



# Create application
app = Flask(__name__)
pickle_in = open("model_lr_new.pkl","rb")
model_lr=pickle.load(pickle_in)
#print(model_lr)
# Bind home function to URL
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model_lr.predict(final_features)

    
    return render_template('index.html', prediction_text='The weight of the fish {}'.format(prediction))





if __name__ == '__main__':
#Run the application
    app.run()
