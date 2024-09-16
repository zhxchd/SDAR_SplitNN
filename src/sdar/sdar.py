import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
from keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from models.attacker_models import make_decoder, make_simulator_discriminator, make_decoder_discriminator
from models.plainnet import make_f as make_diff_simulator

class SDARAttacker:
    def __init__(self, client_ds, server_ds, num_class=10, batch_size=128) -> None:
        self.client_ds = client_ds
        self.server_ds = server_ds
        self.num_class = num_class
        self.batch_size = batch_size
        self.client_dataset = client_ds.repeat(-1).batch(batch_size, drop_remainder=True)
        self.server_dataset = server_ds.repeat(-1).batch(batch_size, drop_remainder=True)
    
    @tf.function
    def fg_train_step(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            z = self.f(x, training=True)
            y_pred = self.g(z, training=True)
            if self.u_shape:
                y_pred = self.h(y_pred, training=True)
            if self.alpha > 0.0:
                fg_loss = sparse_categorical_crossentropy(y, y_pred, from_logits=True) * (1-self.alpha) + self.alpha * dist_corr(x, z)
            else:
                fg_loss = sparse_categorical_crossentropy(y, y_pred, from_logits=True)
            self.fg_acc.update_state(y, y_pred)

        # update f and g
        if not self.u_shape:
            fg_grads = tape.gradient(fg_loss, self.f.trainable_variables + self.g.trainable_variables)
            self.fg_optimizer.apply_gradients(zip(fg_grads, self.f.trainable_variables + self.g.trainable_variables))
        else:
            fgh_grads = tape.gradient(fg_loss, self.f.trainable_variables + self.g.trainable_variables + self.h.trainable_variables)
            self.fg_optimizer.apply_gradients(zip(fgh_grads, self.f.trainable_variables + self.g.trainable_variables + self.h.trainable_variables))
        
        return fg_loss, z

    @tf.function
    def e_train_step(self, xs, ys, y, z):
        # update f_simulator (and h_simulator if u-shaped)
        with tf.GradientTape(persistent=True) as tape:
            zs = self.e(xs, training=True)
            y_pred_simulator = self.g(zs, training=False)
            if self.u_shape:
                y_pred_simulator = self.hs(y_pred_simulator, training=True)
            eg_loss = sparse_categorical_crossentropy(ys, y_pred_simulator, from_logits=True)
            self.eg_acc.update_state(ys, y_pred_simulator)
            # discriminator distinguish zs and z
            if self.e_dis != None:
                e_dis_fake_output = self.e_dis([zs,ys], training=True) if self.conditional else self.e_dis(zs, training=True)
                e_dis_real_output = self.e_dis([z,y], training=True) if self.conditional else self.e_dis(z, training=True)
                e_dis_real_loss = binary_crossentropy(tf.ones_like(e_dis_real_output), e_dis_real_output, from_logits=True)
                self.e_dis_real_acc.update_state(tf.ones_like(e_dis_real_output), e_dis_real_output)
                e_dis_fake_loss = binary_crossentropy(tf.zeros_like(e_dis_fake_output), e_dis_fake_output, from_logits=True)
                self.e_dis_fake_acc.update_state(tf.zeros_like(e_dis_fake_output), e_dis_fake_output)
                e_dis_loss = e_dis_real_loss + e_dis_fake_loss
                e_gen_loss = binary_crossentropy(tf.ones_like(e_dis_fake_output), e_dis_fake_output, from_logits=True)
                if self.alpha > 0.0:
                    e_loss = eg_loss * (1-self.alpha) + e_gen_loss * self.config["lambda1"] + self.alpha*dist_corr(xs, zs)
                else:
                    e_loss = eg_loss + e_gen_loss * self.config["lambda1"]
            else:
                e_dis_real_loss = 0.0
                e_dis_fake_loss = 0.0
                e_dis_loss = 0.0
                e_gen_loss = 0.0
                e_loss = eg_loss
            if self.u_shape:
                hs_loss = eg_loss

        # update e (simulator)
        e_grads = tape.gradient(e_loss, self.e.trainable_variables)
        self.e_optimizer.apply_gradients(zip(e_grads, self.e.trainable_variables))

        if self.e_dis != None:
            # update e_dis
            e_dis_grads = tape.gradient(e_dis_loss, self.e_dis.trainable_variables)
            self.e_dis_optimizer.apply_gradients(zip(e_dis_grads, self.e_dis.trainable_variables))

        if self.u_shape:
            # update h_simulator
            hs_grads = tape.gradient(hs_loss, self.hs.trainable_variables)
            self.hs_optimizer.apply_gradients(zip(hs_grads, self.hs.trainable_variables))
        
        return eg_loss, e_gen_loss, e_loss, e_dis_real_loss, e_dis_fake_loss, e_dis_loss

    @tf.function
    def d_train_step(self, xs, ys, y, z):
        zs = self.e(xs, training=True) # decode the updated zs
        with tf.GradientTape(persistent=True) as tape:
            decoded_xs = self.d([zs,ys], training=True) if self.conditional else self.d(zs, training=True)
            d_mse_loss = tf.reduce_mean(tf.square(xs - decoded_xs), axis=[1,2,3])
            if self.d_dis != None:
                # discriminator distinguish decoded_x and xs
                decoded_x = self.d([z,y], training=True) if self.conditional else self.d(z, training=True)
                d_dis_fake_output = self.d_dis([decoded_x, y], training=True) if self.conditional else self.d_dis(decoded_x, training=True)
                d_dis_real_output = self.d_dis([xs,ys], training=True) if self.conditional else self.d_dis(xs, training=True)
                d_dis_real_loss = binary_crossentropy(tf.ones_like(d_dis_real_output), d_dis_real_output, from_logits=True)
                self.d_dis_real_acc.update_state(tf.ones_like(d_dis_real_output), d_dis_real_output)
                d_dis_fake_loss = binary_crossentropy(tf.zeros_like(d_dis_fake_output), d_dis_fake_output, from_logits=True)
                self.d_dis_fake_acc.update_state(tf.zeros_like(d_dis_fake_output), d_dis_fake_output)
                d_dis_loss = d_dis_real_loss + d_dis_fake_loss
                d_gen_loss = binary_crossentropy(tf.ones_like(d_dis_fake_output), d_dis_fake_output, from_logits=True)
                d_loss = d_mse_loss + d_gen_loss * self.config["lambda2"]
            else:
                d_dis_loss = 0.0
                d_gen_loss = 0.0
                d_dis_real_loss = 0.0
                d_dis_fake_loss = 0.0
                d_loss = d_mse_loss
        # update d
        d_grads = tape.gradient(d_loss, self.d.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.d.trainable_variables))
        if self.d_dis != None:
            # update d_dis
            d_dis_grads = tape.gradient(d_dis_loss, self.d_dis.trainable_variables)
            self.d_dis_optimizer.apply_gradients(zip(d_dis_grads, self.d_dis.trainable_variables))
        return d_mse_loss, d_gen_loss, d_loss, d_dis_real_loss, d_dis_fake_loss, d_dis_loss

    def run(self, level, num_iter, config, fg_lr=0.001, u_shape=False, conditional=True, use_e_dis=True, use_d_dis=True, model_type="resnet", width="standard", diff_simulator=False, dropout=0.0, l1=0.0, l2=0.0, alpha=0.0, verbose_freq=100):
        # check arguments validity
        if u_shape and conditional:
            raise ValueError("Cannot use both u-shaped splitNN and conditional GAN, since there are no labels in U-shaped SL.")
        if width not in {"narrow", "standard", "wide"}:
            raise ValueError(f"Invalid width {width}, must be one of 'narrow', 'standard', 'wide'.")
        if diff_simulator and model_type == "plainnet":
            raise ValueError("Cannot use different simulator with plainnet. Use resnet instead.")
        
        if verbose_freq is not None:
            print("Start running.")
        self.log = {}
        self.log["fg_loss"] = np.zeros(num_iter)
        self.log["fg_acc"] = np.zeros(num_iter)
        self.log["eg_loss"] = np.zeros(num_iter)
        self.log["eg_acc"] = np.zeros(num_iter)
        self.log["e_gen_loss"] = np.zeros(num_iter)
        self.log["e_loss"] = np.zeros(num_iter)
        self.log["e_dis_real_loss"] = np.zeros(num_iter)
        self.log["e_dis_real_acc"] = np.zeros(num_iter)
        self.log["e_dis_fake_loss"] = np.zeros(num_iter)
        self.log["e_dis_fake_acc"] = np.zeros(num_iter)
        self.log["e_dis_loss"] = np.zeros(num_iter)
        self.log["d_mse_loss"] = np.zeros(num_iter)
        self.log["d_gen_loss"] = np.zeros(num_iter)
        self.log["d_loss"] = np.zeros(num_iter)
        self.log["d_dis_real_loss"] = np.zeros(num_iter)
        self.log["d_dis_real_acc"] = np.zeros(num_iter)
        self.log["d_dis_fake_loss"] = np.zeros(num_iter)
        self.log["d_dis_fake_acc"] = np.zeros(num_iter)
        self.log["d_dis_loss"] = np.zeros(num_iter)
        self.log["attack_loss"] = np.zeros(num_iter)
        if u_shape:
            self.log["label_acc"] = np.zeros(num_iter)
        self.level = level
        self.num_iter = num_iter
        self.u_shape = u_shape
        self.conditional = conditional
        if u_shape and conditional:
            raise ValueError("Cannot use both u-shaped splitNN and conditional GAN.")
        if config["lambda1"] == 0.0:
            use_e_dis = False
        if config["lambda2"] == 0.0:
            use_d_dis = False
        self.use_e_dis = use_e_dis
        self.use_d_dis = use_d_dis
        if "flip_rate" not in config:
            config["flip_rate"] = 0.0
        self.flip_rate = config["flip_rate"]
        self.config = config
        self.alpha = alpha
        self.width = width
        self.model_type = model_type

        if model_type == "resnet":
            from models.resnet import make_f, make_g, make_h
        elif model_type == "plainnet":
            from models.plainnet import make_f, make_g, make_h

        # Initialize models
        self.input_shape = self.client_ds.element_spec[0].shape
        self.f = make_f(self.level, self.input_shape, width=width, l1=l1, l2=l2)
        if diff_simulator:
            self.e = make_diff_simulator(self.level, self.input_shape, width=width, l1=l1, l2=l2)
        else:
            self.e = make_f(self.level, self.input_shape, width=width, l1=l1, l2=l2)
        self.intermidiate_shape = self.f.layers[-1].output_shape[1:]

        self.g = make_g(self.level, self.intermidiate_shape, self.num_class, include_h=self.u_shape, dropout=dropout, width=width, l1=l1, l2=l2)
        if self.u_shape:
            self.h = make_h(num_classes=self.num_class, dropout=dropout, l1=l1, l2=l2)
            self.hs = make_h(num_classes=self.num_class, dropout=dropout, l1=l1, l2=l2)
        if self.use_e_dis:
            self.e_dis = make_simulator_discriminator(level, self.intermidiate_shape, self.conditional, num_class=self.num_class, width=self.width)
        else:
            self.e_dis = None
        if self.use_d_dis:
            self.d_dis = make_decoder_discriminator(self.input_shape, self.conditional, num_class=self.num_class)
        else:
            self.d_dis = None
        self.d = make_decoder(self.level, self.intermidiate_shape, self.conditional, num_class=self.num_class, width=self.width)

        # Initialize optimizers
        self.fg_optimizer = Adam(learning_rate=fg_lr)
        self.e_optimizer = Adam(learning_rate=config["e_lr"])
        if self.e_dis != None:
            self.e_dis_optimizer = Adam(learning_rate=config["e_dis_lr"])
        self.d_optimizer = Adam(learning_rate=config["d_lr"])
        if self.d_dis != None:
            self.d_dis_optimizer = Adam(learning_rate=config["d_dis_lr"])
        if self.u_shape:
            self.hs_optimizer = Adam(learning_rate=config["hs_lr"])

        # Initialize metrics
        self.fg_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.eg_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.e_dis_real_acc = tf.keras.metrics.BinaryAccuracy()
        self.e_dis_fake_acc = tf.keras.metrics.BinaryAccuracy()
        self.d_dis_real_acc = tf.keras.metrics.BinaryAccuracy()
        self.d_dis_fake_acc = tf.keras.metrics.BinaryAccuracy()
        
        # start training and attacking
        iterator = zip(self.client_dataset.take(num_iter), self.server_dataset.take(num_iter))

        for i, ((x,y), (xs,ys)) in enumerate(iterator):
            if self.flip_rate > 0.0:
                ys = tf.random.categorical(tf.math.log(tf.one_hot(tf.reshape(ys, ys.shape[0]), self.num_class) * (1-self.flip_rate) + self.flip_rate/self.num_class), 1)
                ys = tf.reshape(ys, (ys.shape[0],1))

            # reset metrics
            self.fg_acc.reset_states()
            self.eg_acc.reset_states()
            self.e_dis_real_acc.reset_states()
            self.e_dis_fake_acc.reset_states()
            self.d_dis_real_acc.reset_states()
            self.d_dis_fake_acc.reset_states()

            # update f and g
            fg_loss, z = self.fg_train_step(x, y)

            # update e
            eg_loss, e_gen_loss, e_loss, e_dis_real_loss, e_dis_fake_loss, e_dis_loss = self.e_train_step(xs, ys, y, z)

            # update d
            d_mse_loss, d_gen_loss, d_loss, d_dis_real_loss, d_dis_fake_loss, d_dis_loss = self.d_train_step(xs, ys, y, z)

            # in the end, we reconstruct the original image
            x_reconstructed = self.d([z,y], training=False) if self.conditional else self.d(z, training=False)
            attack_loss = tf.reduce_mean(tf.square(x - x_reconstructed), axis=[1,2,3]) # batch_size * 1

            # if u_shaped, we can also give a label prediction
            if self.u_shape:
                y_reconstructed = self.hs(self.g(z, training=False), training=False)
                # calculate the accuracy
                label_acc = tf.keras.metrics.SparseCategoricalAccuracy()(y, y_reconstructed)
        
            self.log["fg_loss"][i] = np.mean(fg_loss)
            self.log["fg_acc"][i] = self.fg_acc.result().numpy()
            self.log["eg_loss"][i] = np.mean(eg_loss)
            self.log["eg_acc"][i] = self.eg_acc.result().numpy()
            self.log["e_gen_loss"][i] = np.mean(e_gen_loss)
            self.log["e_loss"][i] = np.mean(e_loss)
            self.log["e_dis_real_loss"][i] = np.mean(e_dis_real_loss)
            self.log["e_dis_real_acc"][i] = self.e_dis_real_acc.result().numpy()
            self.log["e_dis_fake_loss"][i] = np.mean(e_dis_fake_loss)
            self.log["e_dis_fake_acc"][i] = self.e_dis_fake_acc.result().numpy()
            self.log["e_dis_loss"][i] = np.mean(e_dis_loss)
            self.log["d_mse_loss"][i] = np.mean(d_mse_loss)
            self.log["d_gen_loss"][i] = np.mean(d_gen_loss)
            self.log["d_loss"][i] = np.mean(d_loss)
            self.log["d_dis_real_loss"][i] = np.mean(d_dis_real_loss)
            self.log["d_dis_real_acc"][i] = self.d_dis_real_acc.result().numpy()
            self.log["d_dis_fake_loss"][i] = np.mean(d_dis_fake_loss)
            self.log["d_dis_fake_acc"][i] = self.d_dis_fake_acc.result().numpy()
            self.log["d_dis_loss"][i] = np.mean(d_dis_loss)
            self.log["attack_loss"][i] = np.mean(attack_loss)
            
            if self.u_shape:
                self.log["label_acc"][i] = np.mean(label_acc)

            if verbose_freq is not None and (i+1) % verbose_freq == 0:
                # print the following: fg_loss, f_simulator_loss, simulator_discriminator_loss, decoder_total_loss, decoder_discriminator_loss, attack_loss
                if not self.u_shape:
                    print(f"[{i}]: fg_loss: {np.mean(self.log['fg_loss'][i+1-verbose_freq:i+1]):.4f}, e_total_loss: {np.mean(self.log['e_loss'][i+1-verbose_freq:i+1]):.4f}, e_dis_loss: {np.mean(self.log['e_dis_loss'][i+1-verbose_freq:i+1]):.4f}, d_total_loss: {np.mean(self.log['d_loss'][i+1-verbose_freq:i+1]):.4f}, d_dis_loss: {np.mean(self.log['d_dis_loss'][i+1-verbose_freq:i+1]):.4f}, attack_loss: {np.mean(self.log['attack_loss'][i+1-verbose_freq:i+1]):.4f}")
                if self.u_shape:
                    print(f"[{i}]: fg_loss: {np.mean(self.log['fg_loss'][i+1-verbose_freq:i+1]):.4f}, e_total_loss: {np.mean(self.log['e_loss'][i+1-verbose_freq:i+1]):.4f}, e_dis_loss: {np.mean(self.log['e_dis_loss'][i+1-verbose_freq:i+1]):.4f}, d_total_loss: {np.mean(self.log['d_loss'][i+1-verbose_freq:i+1]):.4f}, d_dis_loss: {np.mean(self.log['d_dis_loss'][i+1-verbose_freq:i+1]):.4f}, attack_loss: {np.mean(self.log['attack_loss'][i+1-verbose_freq:i+1]):.4f}, label_acc: {np.mean(self.log['label_acc'][i+1-verbose_freq:i+1]):.4f}")
        return self.log
    
    # attack a single batch
    def attack(self, x, y):
        x_recon = self.d([self.f(x, training=True), y], training=False) if self.conditional else self.d(self.f(x, training=True), training=False)
        return x_recon, np.mean(tf.reduce_mean(tf.square(x - x_recon), axis=[1,2,3]))
    
    # evaluate the attacker, the average MSE for all batches of client's data
    def evaluate(self, verbose=False):
        eval_ds = self.client_ds.batch(self.batch_size, drop_remainder=True)
        iterator = iter(eval_ds)
        total_mse = 0.0
        total_ssim = 0.0
        count = 0
        if self.u_shape:
            total_acc = 0.0
        for (x,y) in iterator:
            z = self.f(x, training=True)
            x_recon = self.d([z,y], training=False) if self.conditional else self.d(z, training=False)
            total_mse += np.mean(tf.reduce_mean(tf.square(x - x_recon), axis=[1,2,3]))
            total_ssim += np.mean(tf.image.ssim(x, x_recon, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03))
            if self.u_shape:
                total_acc += np.mean(tf.keras.metrics.SparseCategoricalAccuracy()(y, self.hs(self.g(z, training=False), training=False)))
            count += 1
        if verbose:
            print(f"Average MSE over all client's images: {total_mse / count}.")
            print(f"Average SSIM over all client's images: {total_ssim / count}.")
            if self.u_shape:
                print(f"Average label accuracy over all client's images: {total_acc / count}.")
        if self.u_shape:
            return total_mse / count, total_ssim/count, total_acc / count
        else:
            return total_mse / count, total_ssim/count
        
def pairwise_dist(A):
    r = tf.reduce_sum(A*A, 1)
    r = tf.reshape(r, [-1, 1])
    D = tf.maximum(r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r), 1e-7)
    D = tf.sqrt(D+1e-10)
    return D

def dist_corr(X, Y):
    X = tf.keras.layers.Flatten()(X)
    Y = tf.keras.layers.Flatten()(Y)
    n = tf.cast(tf.shape(X)[0], tf.float32)
    a = pairwise_dist(X)
    b = pairwise_dist(Y)
    A = a - tf.reduce_mean(a, axis=1) -\
        tf.expand_dims(tf.reduce_mean(a,axis=0),axis=1)+\
        tf.reduce_mean(a)
    B = b - tf.reduce_mean(b, axis=1) -\
        tf.expand_dims(tf.reduce_mean(b,axis=0),axis=1)+\
        tf.reduce_mean(b)
    dCovXY = tf.sqrt(1e-10+tf.reduce_sum(A*B) / (n ** 2))
    dVarXX = tf.sqrt(1e-10+tf.reduce_sum(A*A) / (n ** 2))
    dVarYY = tf.sqrt(1e-10+tf.reduce_sum(B*B) / (n ** 2))

    dCorXY = dCovXY / tf.sqrt(1e-10 + dVarXX * dVarYY)
    return dCorXY