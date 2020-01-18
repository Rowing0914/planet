import tensorflow as tf


def eager_setup():
    """
    it enables an eager execution in TensorFlow with config that allows us to flexibly access to a GPU
    from multiple python scripts
    """
    # === before TF 2.0 ===
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    tf.compat.v1.enable_eager_execution(config=config)
    tf.compat.v1.enable_resource_variables()

    # === For TF 2.0 ===
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # tf.compat.v1.InteractiveSession(config=config)
