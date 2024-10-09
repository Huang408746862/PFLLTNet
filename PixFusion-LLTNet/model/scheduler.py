import tensorflow as tf
import math

class CosineDecayWithRestartsLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, min_lr, first_decay_steps, t_mul=1.5, m_mul=0.95):
        super(CosineDecayWithRestartsLearningRateSchedule, self).__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = min_lr / initial_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        first_decay_steps = tf.cast(self.first_decay_steps, tf.float32)
        t_mul = tf.cast(self.t_mul, tf.float32)
        step = tf.maximum(step, 0)
        first_decay_steps = tf.maximum(first_decay_steps, 1.0)
        i_restart = tf.floor(tf.math.log1p(step / first_decay_steps) / tf.math.log(t_mul))
        sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
        steps_since_restart = step - sum_r * first_decay_steps
        decay_steps = tf.maximum(first_decay_steps * t_mul ** i_restart, 1.0)
        completed_fraction_since_restart = steps_since_restart / decay_steps
        cosine_decay = 0.5 * (1.0 + tf.math.cos(math.pi * tf.clip_by_value(completed_fraction_since_restart, 0.0, 1.0)))
        decayed = (1.0 - self.alpha) * cosine_decay + self.alpha
        new_lr = self.initial_lr * decayed * (self.m_mul ** i_restart)
        return tf.maximum(new_lr, self.min_lr)
