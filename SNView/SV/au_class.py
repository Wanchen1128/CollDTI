import numpy as np
import tensorflow as tf

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class Autoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.keras.optimizers.Adam(1e-3), scale=0.2):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        # self.scale = tf.placeholder(tf.float32)
        
        tf.compat.v1.disable_eager_execution()
        self.scale = tf.compat.v1.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        tf.compat.v1.disable_eager_execution()
        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.n_input])
        self.noisex = self.x+scale*tf.random.normal((n_input,))
        
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random.normal((n_input,)),
                                                      self.weights['w1']), self.weights['b1']))


        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights


    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X,
                                                                          self.scale: self.training_scale})
        return cost

    def before_loss(self, X):
        cost = self.sess.run((self.cost), feed_dict={self.x: X,
                                                     self.scale: self.training_scale})
        return cost


    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate(self, hidden=None):
        if hidden is None:
            # print(self.weights["b1"].shape)
            hidden = np.random.normal(size=self.weights["b1"])
            # hidden = np.random.normal(size=self.weights["b1"].shape)

        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})


    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBias(self):
        return self.sess.run(self.weights['b1'])



def xavier_init(fan_in, fan_out):
    limit = tf.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform(shape=(fan_in, fan_out), minval=-limit, maxval=limit)

def xavier_init(fan_in, fan_out):
    limit = tf.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform(shape=(fan_in, fan_out), minval=-limit, maxval=limit)

class Autoencoder2(tf.keras.Model):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.keras.optimizers.Adam(1e-3), scale=0.2):
        super(Autoencoder2, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.training_scale = scale
        self.optimizer = optimizer
        
        self.encoder = tf.keras.layers.Dense(
            n_hidden, 
            activation=transfer_function,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer='zeros',
            name='encoder'
        )
        
        self.decoder = tf.keras.layers.Dense(
            n_input,
            activation=None,
            kernel_initializer='zeros',
            bias_initializer='zeros',
            name='decoder'
        )
        
        self(tf.zeros([1, n_input]), training=False)
    
    def call(self, inputs, training=False):
        if training:
            noisy_inputs = inputs + self.training_scale * tf.random.normal(tf.shape(inputs))
        else:
            noisy_inputs = inputs
            
        hidden = self.encoder(noisy_inputs)
        reconstruction = self.decoder(hidden)
        
        return reconstruction, hidden
    
    def partial_fit(self, X):
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            reconstruction, _ = self(X_tensor, training=True)
            cost = 0.5 * tf.reduce_sum(tf.square(reconstruction - X_tensor))

        gradients = tape.gradient(cost, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return cost.numpy()
    
    def before_loss(self, X):
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        reconstruction, _ = self(X_tensor, training=False)
        cost = 0.5 * tf.reduce_sum(tf.square(reconstruction - X_tensor))
        return cost.numpy()
    
    def transform(self, X):
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        _, hidden = self(X_tensor, training=False)
        return hidden.numpy()
    
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=(1, self.n_hidden))
        
        hidden_tensor = tf.convert_to_tensor(hidden, dtype=tf.float32)
        reconstruction = self.decoder(hidden_tensor)
        return reconstruction.numpy()
    
    def reconstruct(self, X):
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        reconstruction, _ = self(X_tensor, training=False)
        return reconstruction.numpy()
    
    def getWeights(self):
        return self.encoder.weights[0].numpy()
    
    def getBias(self):
        return self.encoder.weights[1].numpy()