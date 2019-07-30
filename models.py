from keras.engine import Layer
from keras.models import Model
from keras.layers import K, Dense, Input, CuDNNLSTM, Bidirectional, Activation, CuDNNGRU, Embedding, concatenate
from keras.layers import Dropout, GlobalMaxPooling1D, Flatten, GlobalAveragePooling1D, SpatialDropout1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

# Capsule Layer implemented using Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        print(b)
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

                    
def CapsuleNetwork(embedding_matrix,
                   max_len=150, 
                   max_features=100000, 
                   embed_size=300,
                   train_embedding=False,
                   spatial_dropout_rate=0.4, 
                   gru_units=128, 
                   num_capsule=10, 
                   dim_capsule=16, 
                   routings=5, 
                   dropout_rate=0.25,
                   max_pool=False,
                   num_class=6, 
                   act='sigmoid'):
    
    input1_pre = Input(shape=(max_len,))
    embed_layer1_pre = Embedding(max_features,
                            embed_size,
                            input_length=max_len,
                            weights=[embedding_matrix],
                            trainable=train_embedding)(input1_pre)
    embed_layer1_pre = SpatialDropout1D(spatial_dropout_rate)(embed_layer1_pre)
    
    x_pre = Bidirectional(CuDNNGRU(gru_units, return_sequences=True))(embed_layer1_pre)
    capsule_pre = Capsule(num_capsule=num_capsule, dim_capsule=dim_capsule, routings=routings, share_weights=True)(x_pre)
    # capsule_pre = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule_pre)
    
    if max_pool:   
        capsule_pre = GlobalMaxPooling1D()(capsule_pre)
    else:
        capsule_pre = Flatten()(capsule_pre)
    capsule_pre = Dropout(dropout_rate)(capsule_pre)
    
    input1_post = Input(shape=(max_len,))
    embed_layer1_post = Embedding(max_features,
                            embed_size,
                            input_length=max_len,
                            weights=[embedding_matrix],
                            trainable=train_embedding)(input1_post)
    embed_layer1_post = SpatialDropout1D(spatial_dropout_rate)(embed_layer1_post)
    
    x_post = Bidirectional(CuDNNGRU(gru_units, return_sequences=True))(embed_layer1_post)
    capsule_post = Capsule(num_capsule=num_capsule, dim_capsule=dim_capsule, routings=routings, share_weights=True)(x_post)
    # capsule_post = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule_post)
    
    if max_pool:   
        capsule_post = GlobalMaxPooling1D()(capsule_post)
    else:
        capsule_post = Flatten()(capsule_post)
    capsule_post = Dropout(dropout_rate)(capsule_post)
    
    concat = concatenate([capsule_pre,capsule_post])
    output = Dense(num_class, activation=act)(concat)
    
    model = Model(inputs=[input1_pre, input1_post], outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=1e-3, decay=0),
        metrics=['accuracy'])
    
    print (model.summary())
    return model

def CapsuleNetwork(embedding_matrix,
                   max_len=150, 
                   max_features=100000, 
                   embed_size=300,
                   train_embedding=False,
                   spatial_dropout_rate=0.4, 
                   gru_units=128, 
                   num_capsule=10, 
                   dim_capsule=16, 
                   routings=5, 
                   dropout_rate=0.25,
                   max_pool=False,
                   num_class=6, 
                   act='sigmoid'):
    
    input1_pre = Input(shape=(max_len,))
    embed_layer1_pre = Embedding(max_features,
                            embed_size,
                            input_length=max_len,
                            weights=[embedding_matrix],
                            trainable=train_embedding)(input1_pre)
    embed_layer1_pre = SpatialDropout1D(spatial_dropout_rate)(embed_layer1_pre)
    
    x_pre = Bidirectional(CuDNNGRU(gru_units, return_sequences=True))(embed_layer1_pre)
    capsule_pre = Capsule(num_capsule=num_capsule, dim_capsule=dim_capsule, routings=routings, share_weights=True)(x_pre)
    # capsule_pre = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule_pre)
    
    if max_pool:   
        capsule_pre = GlobalMaxPooling1D()(capsule_pre)
    else:
        capsule_pre = Flatten()(capsule_pre)
    capsule_pre = Dropout(dropout_rate)(capsule_pre)
    
    input1_post = Input(shape=(max_len,))
    embed_layer1_post = Embedding(max_features,
                            embed_size,
                            input_length=max_len,
                            weights=[embedding_matrix],
                            trainable=train_embedding)(input1_post)
    embed_layer1_post = SpatialDropout1D(spatial_dropout_rate)(embed_layer1_post)
    
    x_post = Bidirectional(CuDNNGRU(gru_units, return_sequences=True))(embed_layer1_post)
    capsule_post = Capsule(num_capsule=num_capsule, dim_capsule=dim_capsule, routings=routings, share_weights=True)(x_post)
    # capsule_post = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule_post)
    
    if max_pool:   
        capsule_post = GlobalMaxPooling1D()(capsule_post)
    else:
        capsule_post = Flatten()(capsule_post)
    capsule_post = Dropout(dropout_rate)(capsule_post)
    
    concat = concatenate([capsule_pre,capsule_post])
    output = Dense(num_class, activation=act)(concat)
    
    model = Model(inputs=[input1_pre, input1_post], outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=1e-3, decay=0),
        metrics=['accuracy'])
    
    print (model.summary())
    return model


from keras.layers import K, Activation
from keras.engine import Layer
from keras.layers import Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D

class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config
    
def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv1D(filters=dim_capsule*n_channels, kernel_size=kernel_size)(inputs)
    print(output.shape)
#     outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash)(output)

class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        inputs_expand = K.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, dim=1)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                b += K.batch_dot(outputs, inputs_hat, [2, 3])
        # End: Routing algorithm -----------------------------------------------------------------------#

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

def CapsuleNetworkLarge(embedding_matrix,
                   max_len=150, 
                   max_features=100000, 
                   embed_size=300,
                   train_embedding=False,
                   spatial_dropout_rate=0.4, 
                   gru_units=128, 
                   dropout_rate=0.25,
                   num_class=6, 
                   act='sigmoid'):

    input1_pre = Input(shape=(max_len,))
    embed_layer1_pre = Embedding(max_features,
                            embed_size,
                            input_length=max_len,
                            weights=[embedding_matrix],
                            trainable=train_embedding)(input1_pre)
    embed_layer1_pre = SpatialDropout1D(spatial_dropout_rate)(embed_layer1_pre)

    x_pre = Bidirectional(CuDNNGRU(gru_units, return_sequences=True))(embed_layer1_pre)
    capsule_pre1 = PrimaryCap(x_pre,16,1,1)
    capsule_pre2 = PrimaryCap(x_pre,16,1,3)
    capsule_pre3 = PrimaryCap(x_pre,16,1,5)
    toxiccaps_pre1 = CapsuleLayer(num_capsule=6, dim_capsule=16, routings=7, name='toxiccapspre1')(capsule_pre1)
    toxiccaps_pre2 = CapsuleLayer(num_capsule=6, dim_capsule=16, routings=7, name='toxiccapspre2')(capsule_pre2)
    toxiccaps_pre3 = CapsuleLayer(num_capsule=6, dim_capsule=16, routings=7, name='toxiccapspre3')(capsule_pre3)
    toxiccaps_pre = concatenate([toxiccaps_pre1,toxiccaps_pre2,toxiccaps_pre3])
    capsule_pre = GlobalMaxPooling1D()(toxiccaps_pre)
    capsule_pre = Dropout(dropout_rate)(capsule_pre)

    input1_post = Input(shape=(max_len,))
    embed_layer1_post = Embedding(max_features,
                            embed_size,
                            input_length=max_len,
                            weights=[embedding_matrix],
                            trainable=train_embedding)(input1_post)
    embed_layer1_post = SpatialDropout1D(spatial_dropout_rate)(embed_layer1_post)

    x_post = Bidirectional(CuDNNGRU(gru_units, return_sequences=True))(embed_layer1_post)
    capsule_post1 = PrimaryCap(x_post,16,1,1)
    capsule_post2 = PrimaryCap(x_post,16,1,3)
    capsule_post3 = PrimaryCap(x_post,16,1,5)
    toxiccaps_post1 = CapsuleLayer(num_capsule=6, dim_capsule=16, routings=7, name='toxiccapspost1')(capsule_post1)
    toxiccaps_post2 = CapsuleLayer(num_capsule=6, dim_capsule=16, routings=7, name='toxiccapspost2')(capsule_post2)
    toxiccaps_post3 = CapsuleLayer(num_capsule=6, dim_capsule=16, routings=7, name='toxiccapspost3')(capsule_post3)
    toxiccaps_post = concatenate([toxiccaps_post1,toxiccaps_post2,toxiccaps_post3])
    capsule_post = GlobalMaxPooling1D()(toxiccaps_post)
    capsule_post = Dropout(dropout_rate)(capsule_post)

    concat = concatenate([capsule_pre, capsule_post])
    output = Dense(num_class, activation=act)(concat)

    model = Model(inputs=[input1_pre,input1_post], outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=1e-3,decay=0),
        metrics=['accuracy'])
    print (model.summary())
    return model
