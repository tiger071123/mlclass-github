import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#import plotly.express as px
#import plotly.io as pio

dna = pd.read_csv("gs://big-dna/all_classifcation_and_seqs_aln.csv")

enc = LabelEncoder()

dnal = []

for i in range(len(dna)):
    temp = []
    for j in range(len(dna["sequence"].values[0])):
        row = dna["sequence"].values[i]
        if row[j] == '-':
            temp.append(0)
        elif row[j] == 'A':
            temp.append(1)
        elif row[j] == 'T':
            temp.append(2)
        elif row[j] == 'C':
            temp.append(3)
        elif row[j] == 'G':
            temp.append(4)
    dnal.append(temp)
    
print(dnal[0])

processed = pd.DataFrame(dnal)

print(processed.head())

nFeatures = len(processed.columns) #number of features
print(nFeatures)

enc_species = enc.fit_transform(dna["species"]) #encode the species starting from 0
nSpecies = enc_species.max()+1  #number of species can be obtained by max+1
print(nSpecies)

X = processed
y = enc_species

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(nFeatures, input_shape=[nFeatures]),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(40, activation="relu"),
    tf.keras.layers.Dense(nSpecies, activation="relu"),
    tf.keras.layers.Softmax()
])

lr = 0.0001
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=['accuracy']
)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)
loss = pd.DataFrame(history.history)['loss']
# pio.renderers.default = "browser"
# px.scatter(loss).show()