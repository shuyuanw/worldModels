import numpy as np
from collections import namedtuple
import json
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4

@tf.function
def sample_vae(vae_mu, vae_logvar):
    sz = vae_mu.shape[1]
    mu_logvar = tf.concat([vae_mu, vae_logvar], axis=1)
    z = tfp.layers.DistributionLambda(lambda theta: tfp.distributions.MultivariateNormalDiag(loc=theta[:, :sz], scale_diag=tf.exp(theta[:, sz:])), dtype=tf.float16)
    return z(mu_logvar)
class MDNRNN(tf.keras.Model):
    def __init__(self):
        super(MDNRNN, self).__init__()
        #args = args
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
        
        self.loss_fn = self.get_loss() 
        self.inference_base = tf.keras.layers.LSTM(units=256, return_sequences=True, return_state=True, time_major=False)
        rnn_out_size = 5 * 512 * 3 + 0 + 0 # 3 comes from pi, mu, std 
        self.out_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=256),
            tf.keras.layers.Dense(rnn_out_size, name="mu_logstd_logmix_net")])
        super(MDNRNN, self).build((5, 1000, 515))
    def get_loss(self):
        num_mixture = 5
        batch_size = 5
        z_size = 512
        d_true_weight = 1.0
        
        """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
        # Construct a loss function with the right number of mixtures and outputs
        def z_loss_func(y_true, y_pred):
            '''
            This loss function is defined for N*k components each containing a gaussian of 1 feature
            '''
            mdnrnn_params = y_pred
            y_true = tf.reshape(y_true, [batch_size, -1, z_size + 1]) # +1 for mask
            z_true, mask = y_true[:, :, :-1], y_true[:, :, -1:]
            # Reshape inputs in case this is used in a TimeDistribued layer
            mdnrnn_params = tf.reshape(mdnrnn_params, [-1, 3*num_mixture], name='reshape_ypreds')
            vae_z, mask = tf.reshape(z_true, [-1, 1]), tf.reshape(mask, [-1, 1])
            
            out_mu, out_logstd, out_logpi = tf.split(mdnrnn_params, num_or_size_splits=3, axis=1, name='mdn_coef_split')
            out_logpi = out_logpi - tf.reduce_logsumexp(input_tensor=out_logpi, axis=1, keepdims=True) # normalize
            logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
            lognormal = -0.5 * ((vae_z - out_mu) / tf.exp(out_logstd)) ** 2 - out_logstd - logSqrtTwoPI
            v = out_logpi + lognormal
            
            z_loss = -tf.reduce_logsumexp(input_tensor=v, axis=1, keepdims=True)
            mask = tf.reshape(tf.tile(mask, [1, z_size]), [-1, 1]) # tile b/c we consider z_loss is flattene
            z_loss = mask * z_loss # don't train if episode ends
            z_loss = tf.reduce_sum(z_loss) / tf.reduce_sum(mask) 
            return z_loss
        def d_loss_func(y_true, y_pred):
            d_pred = y_pred
            y_true = tf.reshape(y_true, [batch_size, -1, 1 + 1]) # b/c tf is stupid
            d_true, mask = y_true[:, :, :-1], y_true[:, :, -1:]
            d_true, mask = tf.reshape(d_true, [-1, 1]), tf.reshape(mask, [-1, 1])
           
            d_loss = tf.nn.weighted_cross_entropy_with_logits(labels=d_true, logits=d_pred, pos_weight=d_true_weight) 
            d_loss = mask * d_loss
            d_loss = tf.reduce_sum(d_loss) / tf.reduce_sum(mask) # mean of unmasked 
            return d_loss
        def r_loss_func(y_true, y_pred):
            r_pred = y_pred
            y_true = tf.reshape(y_true, [batch_size, -1, 1 + 1]) # b/c tf is stupid
            r_true, mask = y_true[:, :, :-1], y_true[:, :, -1:]
            r_true, mask = tf.reshape(r_true, [-1, 1]), tf.reshape(mask, [-1, 1])
            r_loss = tf.expand_dims(tf.keras.losses.MSE(y_true=r_true, y_pred=r_pred), axis=-1)
            r_loss = mask * r_loss
            r_loss = tf.reduce_sum(r_loss) / tf.reduce_sum(mask)
            return r_loss
        losses = {'MDN': z_loss_func}
        if 0 == 1:
            losses['r'] = r_loss_func
        if 0 == 1:
            losses['d'] = d_loss_func
        return losses
    def set_random_params(self, stdev=0.5):
        params = self.get_weights()
        rand_params = []
        for param_i in params:
            # David's spicy initialization scheme is wild but from preliminary experiments is critical
            sampled_param = np.random.standard_cauchy(param_i.shape)*stdev / 10000.0 
            rand_params.append(sampled_param) # spice things up
          
        self.set_weights(rand_params)
   
    def parse_rnn_out(self, out):
        mdnrnn_param_width = 5 * 512 * 3 # 3 comes from pi, mu, std 
        mdnrnn_params = out[:, :mdnrnn_param_width]
        if 0 == 1:
            r = out[:, mdnrnn_param_width:mdnrnn_param_width+0]
        else:
            r = None
        if 0 == 1:
            d_logits = out[:, mdnrnn_param_width+0:]
        else:
            d_logits = None
        return mdnrnn_params, r, d_logits
    def call(self, inputs, training=True):
        return self.__call__(inputs, training)
    def __call__(self, inputs, training=True):
        rnn_out, _, _ = self.inference_base(inputs, training=training)
        rnn_out = tf.reshape(rnn_out, [-1, 256])
        out = self.out_net(rnn_out)
        mdnrnn_params, r, d_logits = self.parse_rnn_out(out)
       
        outputs = {'MDN': mdnrnn_params} # can't output None b/c tfkeras redirrects to loss for optimization 
        if 0 == 1:
            outputs['r'] = r
        if 0 == 1:
            outputs['d'] = d_logits
        return outputs
@tf.function
def rnn_next_state(rnn, z, a, prev_state):
    z = tf.cast(tf.reshape(z, [1, 1, -1]), tf.float32)
    a = tf.cast(tf.reshape(a, [1, 1, -1]), tf.float32)
    z_a = tf.concat([z, a], axis=2)
    _, h, c = rnn.inference_base(z_a, initial_state=prev_state, training=False) # set training False to NOT use Dropout
    return [h, c]
@tf.function
def rnn_init_state(rnn):
  return rnn.inference_base.cell.get_initial_state(batch_size=1, dtype=tf.float32) 
def rnn_output(state, z, mode):
  state_h, state_c = state[0], state[1]
  if mode == MODE_ZCH:
    return np.concatenate([z, np.concatenate((state_c,state_h), axis=1)[0]])
  if mode == MODE_ZC:
    return np.concatenate([z, state_c[0]])
  if mode == MODE_ZH:
    return np.concatenate([z, state_h[0]])
  return z # MODE_Z or MODE_Z_HIDDEN
