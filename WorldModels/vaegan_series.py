import numpy as np
import os
import json
import tensorflow as tf
import random
from vaegan import encoder, generator
from utils import PARSER
import tensorflow.keras as keras

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
    
DATA_DIR = "results/WorldModels/CarRacing-v0/record"
SERIES_DIR = "results/WorldModels/CarRacing-v0/vaegan_series"
model_path_name = "results/{WorldModels/CarRacing-v0/tf_vaegan"

def ds_gen():
    filenames = os.listdir(DATA_DIR)[:10000] # only use first 10k episodes
    n = len(filenames)
    for j, fname in enumerate(filenames):
        if not fname.endswith('npz'): 
            continue
        file_path = os.path.join(DATA_DIR, fname)

        data = np.load(file_path)
        img = data['obs']
        action = np.reshape(data['action'], newshape=[-1, 3])
        reward = data['reward']
        done = data['done']
        N = data['N']
        
        n_pad = 1000 - img.shape[0] # pad so they are all a thousand step long episodes
        img = tf.pad(img, [[0, n_pad], [0, 0], [0, 0], [0, 0]])
        action = tf.pad(action, [[0, n_pad], [0, 0]])
        reward = tf.pad(reward, [[0, n_pad]])
        done = tf.pad(done, [[0, n_pad]], constant_values=done[-1])
        N = tf.pad(N, [[0, n_pad]], constant_values=N[-1])
        yield img, action, reward, done, N
        
def create_tf_dataset():
    dataset = tf.data.Dataset.from_generator(ds_gen, output_types=(tf.float32, tf.float32, tf.float32, tf.bool, tf.uint16), output_shapes=((1000, 64, 64, 3), (1000, 3), (1000,), (1000,), (1000,)))
    return dataset
    
lr = 0.0001
E_opt = keras.optimizers.Adam(lr=lr)
G_opt = keras.optimizers.Adam(lr=lr)
G = generator()
E = encoder()
#G.compile(optimizer=G_opt)
#E.compile(optimizer=E_opt)
G.load_weights('results/WorldModels/CarRacing-v0/tf_vaegan/generator.h5')
E.load_weights('results/WorldModels/CarRacing-v0/tf_vaegan/encoder.h5')

#@tf.function
def encode_batch(batch_img):
  simple_obs = batch_img/255.0
  new_pred = E.predict(simple_obs)
  return new_pred[2], new_pred[3]
  
def decode_batch(batch_z):
  # decode the latent vector
  batch_img = G.predict(z.reshape(batch_size, z_size)) * 255.
  batch_img = np.round(batch_img).astype(np.uint8)
  batch_img = batch_img.reshape(batch_size, 64, 64, 3)
  return batch_img
  
filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]
dataset = create_tf_dataset()
dataset = dataset.batch(1, drop_remainder=True)

mu_dataset = []
logvar_dataset = []
action_dataset = []
r_dataset = []
d_dataset = []
N_dataset = []

i=0
for batch in dataset:
  i += 1
  obs_batch, action_batch, r, d, N = batch
  obs_batch = tf.squeeze(obs_batch, axis=0)
  action_batch = tf.squeeze(action_batch, axis=0)
  r = tf.reshape(r, [-1, 1])
  d = tf.reshape(d, [-1, 1])
  N = tf.reshape(N, [-1, 1])

  mu, logvar = encode_batch(obs_batch)

  mu_dataset.append(mu.astype(np.float16))
  logvar_dataset.append(logvar.astype(np.float16))
  action_dataset.append(action_batch.numpy())
  r_dataset.append(r.numpy().astype(np.float16))
  d_dataset.append(d.numpy().astype(np.bool))
  N_dataset.append(N.numpy().astype(np.uint16))

  if ((i+1) % 100 == 0):
    print(i+1)
    
action_dataset = np.array(action_dataset)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)
r_dataset = np.array(r_dataset)
d_dataset = np.array(d_dataset)
N_dataset = np.array(N_dataset)

np.savez_compressed(os.path.join(SERIES_DIR, "series.npz"), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset, reward=r_dataset, done=d_dataset, N=N_dataset)
