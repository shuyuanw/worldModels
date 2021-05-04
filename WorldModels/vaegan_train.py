import tensorflow.keras as keras
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import sys
from vaegan import encoder, generator, discriminator, train_step_vaegan

DEPTH = 32
LATENT_DEPTH = 32
K_SIZE = 5
    

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
    
IM_DIM = 64
batch_size = 64
def ds_gen():
    dirname = 'results/WorldModels/CarRacing-v0/record'
    filenames = os.listdir(dirname)[:10000] # only use first 10k episodes
    n = len(filenames)
    for j, fname in enumerate(filenames):
        if not fname.endswith('npz'): 
            continue
        file_path = os.path.join(dirname, fname)
        with np.load(file_path) as data:
            N = data['obs'].shape[0]
            for i, img in enumerate(data['obs']):
                img_i = img / 255.0
                yield img_i

E = encoder()
G = generator()
D = discriminator() 
lr=0.0001
#lr=0.0001
E_opt = keras.optimizers.Adam(lr=lr)
G_opt = keras.optimizers.Adam(lr=lr)
D_opt = keras.optimizers.Adam(lr=lr)

inner_loss_coef = 1
normal_coef = 0.1
kl_coef = 0.01

step = 0
s1 = 0
max_step = 10000000
log_freq,img_log_freq = 10, 100
save_freq,save_number_mult = 1000, 10000

metrics_names = ["gan_loss", "vae_loss", "fake_dis_loss", "r_dis_loss", "t_dis_loss", "vae_inner_loss", "E_loss", "D_loss", "kl_loss", "normal_loss"]
metrics = []
for m in metrics_names :
    metrics.append(tf.keras.metrics.Mean('m', dtype=tf.float32))

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = ('logs/sep_D%dL%d/' % (DEPTH,LATENT_DEPTH)) + current_time 
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
name = ('sep_D%dL%d' % (DEPTH,LATENT_DEPTH))


def print_metrics() :
    s = ""
    for name,metric in zip(metrics_names,metrics) :
        s+= " " + name + " " + str(np.around(metric.result().numpy(), 3)) 
    print(f"\rEpoch : " + str(s1) +" Step : " + str(step) + " " + s, end="", flush=True)
    with train_summary_writer.as_default():
        for name,metric in zip(metrics_names,metrics) :
            tf.summary.scalar(name, metric.result(), step=step)
    for metric in metrics : 
        metric.reset_states()



shuffle_size = 20 * 1000 # only loads ~20 episodes for shuffle windows b/c im poor and don't have much RAM
ds = tf.data.Dataset.from_generator(ds_gen, output_types=tf.float32, output_shapes=(64, 64, 3))
ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=True).batch(64)
ds = ds.prefetch(100) # prefetch 100 batches in the buffer #tf.data.experimental.AUTOTUNE)
    

for i in range(10):
    s1 += 1
    step = 0
    for x in ds:
        step += 1
        if not step % log_freq :
            print_metrics()
        results = train_step_vaegan(x, E, G, D, E_opt, G_opt, D_opt, inner_loss_coef, normal_coef, kl_coef)
        for metric,result in zip(metrics, results) :
            metric(result)
    E.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/encoder.h5')
    G.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/generator.h5')
    D.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/discriminator.h5')

E.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/encoder.h5')
G.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/generator.h5')
D.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/discriminator.h5')
