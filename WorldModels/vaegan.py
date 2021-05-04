import tensorflow.keras as keras
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import sys
'''
DEPTH = 32
LATENT_DEPTH = 512
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
'''

IM_DIM = 64
batch_size = 64
DEPTH = 32
LATENT_DEPTH = 32
K_SIZE = 5

def sampling(args):
    mean, logsigma = args
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))
    return mean + tf.exp(logsigma / 2) * epsilon

def encoder():
    input_E = keras.layers.Input(shape=(IM_DIM, IM_DIM, 3))
    
    X = keras.layers.Conv2D(filters=DEPTH*2, kernel_size=K_SIZE, strides=2, padding='same')(input_E)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2D(filters=DEPTH*4, kernel_size=K_SIZE, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2D(filters=DEPTH*8, kernel_size=K_SIZE, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(LATENT_DEPTH)(X)    
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
    mean = keras.layers.Dense(LATENT_DEPTH,activation="tanh")(X)
    logsigma = keras.layers.Dense(LATENT_DEPTH,activation="tanh")(X)
    latent = keras.layers.Lambda(sampling, output_shape=(LATENT_DEPTH,))([mean, logsigma])
    
    kl_loss = 1 + logsigma - keras.backend.square(mean) - keras.backend.exp(logsigma)
    kl_loss = keras.backend.mean(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    return keras.models.Model(input_E, [latent,kl_loss, mean, logsigma])

def generator():
    input_G = keras.layers.Input(shape=(LATENT_DEPTH,))

    X = keras.layers.Dense(8*8*DEPTH*8)(input_G)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    X = keras.layers.Reshape((8, 8, DEPTH * 8))(X)
    
    X = keras.layers.Conv2DTranspose(filters=DEPTH*8, kernel_size=K_SIZE, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2DTranspose(filters=DEPTH*4, kernel_size=K_SIZE, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
    X = keras.layers.Conv2DTranspose(filters=DEPTH, kernel_size=K_SIZE, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
    X = keras.layers.Conv2D(filters=3, kernel_size=K_SIZE, padding='same')(X)
    X = keras.layers.Activation('sigmoid')(X)

    return keras.models.Model(input_G, X)

def discriminator():
    input_D = keras.layers.Input(shape=(IM_DIM, IM_DIM, 3))
    
    X = keras.layers.Conv2D(filters=DEPTH, kernel_size=K_SIZE, strides=2, padding='same')(input_D)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
    X = keras.layers.Conv2D(filters=DEPTH*4, kernel_size=K_SIZE, strides=2, padding='same')(input_D)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.Conv2D(filters=DEPTH*8, kernel_size=K_SIZE, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2D(filters=DEPTH*8, kernel_size=K_SIZE, padding='same')(X)
    inner_output = keras.layers.Flatten()(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(DEPTH*8)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    
    output = keras.layers.Dense(1)(X)    
    
    return keras.models.Model(input_D, [output, inner_output])
    
'''
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
'''

@tf.function
def train_step_vaegan(x, E, G, D, E_opt, G_opt, D_opt, inner_loss_coef, normal_coef, kl_coef):
    lattent_r =  tf.random.normal((batch_size, LATENT_DEPTH))
    with tf.GradientTape(persistent=True) as tape :
        lattent,kl_loss, m1, l1 = E(x)
        fake = G(lattent)
        dis_fake,dis_inner_fake = D(fake)
        dis_fake_r,_ = D(G(lattent_r))
        dis_true,dis_inner_true = D(x)

        vae_inner = dis_inner_fake-dis_inner_true
        vae_inner = vae_inner*vae_inner
        
        mean,var = tf.nn.moments(E(x)[0], axes=0)
        var_to_one = var - 1
        
        normal_loss = tf.reduce_mean(mean*mean) + tf.reduce_mean(var_to_one*var_to_one)
        
        kl_loss = tf.reduce_mean(kl_loss)
        vae_diff_loss = tf.reduce_mean(vae_inner)
        f_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake), dis_fake))
        r_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake_r), dis_fake_r))
        t_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(dis_true), dis_true))
        gan_loss = (0.5*t_dis_loss + 0.25*f_dis_loss + 0.25*r_dis_loss)
        vae_loss = tf.reduce_mean(tf.abs(x-fake)) 
        E_loss = vae_diff_loss + kl_coef*kl_loss + normal_coef*normal_loss
        G_loss = inner_loss_coef*vae_diff_loss - gan_loss
        D_loss = gan_loss
    
    E_grad = tape.gradient(E_loss,E.trainable_variables)
    G_grad = tape.gradient(G_loss,G.trainable_variables)
    D_grad = tape.gradient(D_loss,D.trainable_variables)
    del tape
    E_opt.apply_gradients(zip(E_grad, E.trainable_variables))
    G_opt.apply_gradients(zip(G_grad, G.trainable_variables))
    D_opt.apply_gradients(zip(D_grad, D.trainable_variables))

    return [gan_loss, vae_loss, f_dis_loss, r_dis_loss, t_dis_loss, vae_diff_loss, E_loss, D_loss, kl_loss, normal_loss]

'''
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
        results = train_step_vaegan(x)
        for metric,result in zip(metrics, results) :
            metric(result)
    E.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/encoder.h5')
    G.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/generator.h5')
    D.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/discriminator.h5')

E.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/encoder.h5')
G.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/generator.h5')
D.save_weights('results/WorldModels/CarRacing-v0/tf_vaegan/discriminator.h5')
'''
            
