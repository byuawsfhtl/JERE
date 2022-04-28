import tensorflow as tf
from sklearn.utils import class_weight

class Tools():
  def __init__(self,y_true):
    self.class_weights=dict(enumerate(list(class_weight.compute_class_weight('balanced', 
                                                     classes=np.unique(y_true),
                                                     y=y_true))))

  def convert_to_logits(y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                              1 - tf.keras.backend.epsilon())
    return tf.math.log(y_pred / (1 - y_pred))

  #faster but less effective
  def weighted_ce_loss(y_true, y_pred):
    beta = 0.9
    y_pred = convert_to_logits(y_pred)
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                    labels=y_true,
                                                    pos_weight=pos_weight)
    return tf.reduce_mean(loss)
    
  #slower but more effective
  def weighted_cce_loss(y_true, y_pred):
    weighted_y_true = tf.cast(y_true*self.class_weights, tf.float32)
    log_prb = -1 * tf.nn.log_softmax(y_pred, axis=1)
    loss = (log_prb * weighted_y_true)
    mean_loss = tf.math.reduce_sum(loss) / tf.math.reduce_sum(weighted_y_true)
    return mean_loss
