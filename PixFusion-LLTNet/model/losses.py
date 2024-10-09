import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import Model


def color_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(tf.reduce_mean(y_true, axis=[1, 2]) - tf.reduce_mean(y_pred, axis=[1, 2])))


def psnr_loss(y_true, y_pred):
    return 40.0 - tf.image.psnr(y_true, y_pred, max_val=1.0 + 1e-5)


def load_vgg():
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return loss_model



def perceptual_loss(y_true, y_pred, loss_model):
    return tf.reduce_mean(tf.square(loss_model(y_true) - loss_model(y_pred)))


def smooth_l1_loss(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    smooth_l1_loss = (0.5 * diff ** 2) * less_than_one + (diff - 0.5) * (1.0 - less_than_one)
    return tf.reduce_mean(smooth_l1_loss)


def multiscale_ssim_loss(y_true, y_pred, max_val=1.0, power_factors=[0.5, 0.5]):
    return 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val, power_factors=power_factors))


def histogram_loss(y_true, y_pred, bins=256):
    y_true_hist = tf.histogram_fixed_width(y_true, [0.0, 1.0], nbins=bins)
    y_pred_hist = tf.histogram_fixed_width(y_pred, [0.0, 1.0], nbins=bins)
    y_true_hist = tf.cast(y_true_hist, tf.float32)
    y_pred_hist = tf.cast(y_pred_hist, tf.float32)
    y_true_hist /= (tf.reduce_sum(y_true_hist) + 1e-5)
    y_pred_hist /= (tf.reduce_sum(y_pred_hist) + 1e-5)
    hist_distance = tf.reduce_mean(tf.abs(y_true_hist - y_pred_hist))
    return hist_distance


def light_consistency_loss(y_true, y_pred):
    true_mean = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
    pred_mean = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)
    return tf.reduce_mean(tf.abs(true_mean - pred_mean))



def contrastive_loss(y_true, y_pred, margin=1.0):
    distance = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    loss = tf.where(distance < margin, 0.5 * tf.square(distance), margin * (distance - 0.5 * margin))

    return tf.reduce_mean(loss)




def loss(y_true, y_pred, loss_model):
    y_true = tf.clip_by_value((y_true + 1.0) / 2.0, 0.0, 1.0)
    y_pred = tf.clip_by_value((y_pred + 1.0) / 2.0, 0.0, 1.0)

    alpha1 = 1.00
    alpha2 = 0.06
    alpha3 = 0.05
    alpha4 = 0.54
    alpha5 = 0.0084
    alpha6 = 0.26
    alpha7 = 0.05
    alpha8 = 0.03

    smooth_l1_l = smooth_l1_loss(y_true, y_pred)
    ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
    perc_l = perceptual_loss(y_true, y_pred, loss_model=loss_model)
    hist_l = histogram_loss(y_true, y_pred)
    psnr_l = psnr_loss(y_true, y_pred)
    color_l = color_loss(y_true, y_pred)
    light_consistency_l = light_consistency_loss(y_true, y_pred)
    contrastive_l = contrastive_loss(y_true, y_pred)  # 新增的对比损失

    total_loss = (alpha1 * smooth_l1_l +
                  alpha2 * perc_l +
                  alpha3 * hist_l +
                  alpha4 * ms_ssim_l +
                  alpha5 * psnr_l +
                  alpha6 * color_l +
                  alpha7 * light_consistency_l +
                  alpha8 * contrastive_l)

    return tf.reduce_mean(total_loss)

