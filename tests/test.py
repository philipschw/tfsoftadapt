from tfsoftadapt import TFSoftAdapt, TFNormalizedSoftAdapt, TFLossWeightedSoftAdapt
import tensorflow as tf
# We redefine the loss components above for the sake of completeness.
loss_component_1 = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0], [150.0, 100.0, 50.0, 10.0, 0.1], [1500.0, 1000.0, 500.0, 100.0, 1.0]])

# Here we define the different SoftAdapt objects
softadapt_object  = TFSoftAdapt(beta=0.1)
normalized_softadapt_object = TFNormalizedSoftAdapt(beta=0.1)
loss_weighted_softadapt_object = TFLossWeightedSoftAdapt(beta=0.1)


softadapt_object.get_component_weights(loss_component_1)
# <tf.Tensor: shape=(3,), dtype=float32, numpy=array([9.9343336e-01, 6.5666283e-03, 3.8908041e-22], dtype=float32)>

normalized_softadapt_object.get_component_weights(loss_component_1)
# <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0.32207233, 0.32507923, 0.35284847], dtype=float32)>

loss_weighted_softadapt_object.get_component_weights(loss_component_1)
# <tf.Tensor: shape=(3,), dtype=float32, numpy=array([8.79777193e-01, 1.20222814e-01, 7.12104538e-20], dtype=float32)>

