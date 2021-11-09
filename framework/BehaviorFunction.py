import tensorflow as tf
import tensorflow.keras.models as M


def prepare_inputs(state, command, to_batch=True):  # inputs arrayLike
    state_tensor = tf.convert_to_tensor(state)
    if to_batch:  # state is not a batch
        state_tensor = tf.expand_dims(state_tensor, 0)

    command_tensor = tf.convert_to_tensor(command)
    if to_batch:  # command is not a batch
        command_tensor = tf.expand_dims(command_tensor, 0)

    return state_tensor, command_tensor


def get_finite_action(model, state, command, training=True):
    inputs = prepare_inputs(state, command)
    probabilities = model(inputs, training)
    categories = tf.random.categorical(probabilities, 1)
    return tf.squeeze(categories, axis=-1)  # tensor1d


def get_finite_greedy_action(model, state, command, training=True):
    inputs = prepare_inputs(state, command)
    probabilities = model(inputs, training)
    return tf.argmax(probabilities, axis=-1)  # tensor1d


class BehaviorFunction(M.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def action(self, state, command, training=True):
        return get_finite_action(self.model, state, command, training)

    def greedy_action(self, state, command, training=True):
        return get_finite_greedy_action(self.model, state, command, training)
