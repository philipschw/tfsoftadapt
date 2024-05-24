"""Unit testing for the original SoftAdapt variant."""

from tfsoftadapt import TFSoftAdapt
import tensorflow as tf
import unittest

class TestSoftAdapt(unittest.TestCase):
    """Class for testing our finite difference implementation."""

    @classmethod
    def setUpClass(class_):
        class_.decimal_place = 5

    # First starting with positive slope test cases.
    def test_beta_positive_three_components(self):
        loss_component1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
        loss_component2 = tf.constant([150, 100, 50, 10, 0.1], dtype=tf.float32)
        loss_component3 = tf.constant([1500, 1000, 500, 100, 1], dtype=tf.float32)

        solutions = tf.constant([9.9343e-01, 6.5666e-03, 3.8908e-22], dtype=tf.float32)

        softadapt_object = TFSoftAdapt(beta=0.1)
        alpha_0, alpha_1, alpha_2 = softadapt_object.get_component_weights(
                                                                loss_component1,
                                                                loss_component2,
                                                                loss_component3,
                                                                verbose=False)
        self.assertAlmostEqual(
            alpha_0.numpy(),
            solutions[0].numpy(),
            self.decimal_place,
            ("Incorrect SoftAdapt calculation for simple 'dominant loss' case."
             "The first loss component failed.")
        )

        self.assertAlmostEqual(
            alpha_1.numpy(),
            solutions[1].numpy(),
            self.decimal_place,
            ("Incorrect SoftAdapt calculation for simple 'dominant loss' case."
             "The second loss component failed.")
        )

        self.assertAlmostEqual(
            alpha_2.numpy(),
            solutions[2].numpy(),
            self.decimal_place,
            ("Incorrect SoftAdapt calculation for simple 'dominant loss' case."
             "The second loss component failed.")
        )


    # TODO: Add more sophisticated unit tests


if __name__ == "__main__":
    unittest.main()
