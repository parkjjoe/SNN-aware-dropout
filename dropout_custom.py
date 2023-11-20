import tensorflow as tf
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.util.tf_export import keras_export


# Original Dropout code
class Dropout(Layer):
  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
    super(Dropout, self).__init__(**kwargs)
    if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
      raise ValueError(f'Invalid value {rate} received for '
                       f'`rate`, expected a value between 0 and 1.')
    self.rate = rate
    self.noise_shape = noise_shape
    self.seed = seed
    self.supports_masking = True

  def _get_noise_shape(self, inputs):
    if self.noise_shape is None:
      return None

    concrete_inputs_shape = array_ops.shape(inputs)
    noise_shape = []
    for i, value in enumerate(self.noise_shape):
      noise_shape.append(concrete_inputs_shape[i] if value is None else value)
    return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()

    def dropped_inputs():
      return nn.dropout(
          inputs,
          noise_shape=self._get_noise_shape(inputs),
          seed=self.seed,
          rate=self.rate)

    output = control_flow_util.smart_cond(training, dropped_inputs,
                                          lambda: array_ops.identity(inputs))
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'rate': self.rate,
        'noise_shape': self.noise_shape,
        'seed': self.seed
    }
    base_config = super(Dropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

# Create a spike based on membrane potential, simultaneously limiting neurons with too many spikes
class Dropout_custom1(Layer):
    def __init__(self, threshold, max_spikes, **kwargs):
        super(Dropout_custom1, self).__init__(**kwargs)
        if not isinstance(max_spikes, int) or max_spikes <0:
            raise ValueError(f'Invalid value {max_spikes} received for '
                             f'`max_spikes`, expected a non-negative integer value.')
        if not 0 <= threshold <= 1:
            raise ValueError(f'Invalid value {threshold} received for '
                             f'`threshold`, expected a value between 0 and 1.')
        self.threshold = threshold
        self.max_spikes = max_spikes
        self.supports_masking = True

    def call(self, x, training=None):
        if training is None:
            training = K.learning_phase()

        # Make a spike(=1) if the threshold is above threshold
        spikes = tf.cast(x > self.threshold, dtype = tf.float32)

        def dropped_inputs(spikes):
            # Restrict the spikes
            spike_counts = tf.reduce_sum(spikes, axis=0) # Calculate the number of spikes of each neurons
            mask = spike_counts < self.max_spikes
            spikes = spikes * tf.cast(mask, dtype=tf.float32)
            return spikes

        output = control_flow_util.smart_cond(training, dropped_inputs, lambda: spikes)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'threshold': self.threshold,
            'max_spikes': self.max_spikes
        }
        base_config = super(Dropout_custom1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Neurons with more spikes than max_spikees are output as 0
class Dropout_custom2(Layer):
    def __init__(self, max_spikes, **kwargs):
        super(Dropout_custom2, self).__init__(**kwargs)
        if not isinstance(max_spikes, int) or max_spikes <0:
            raise ValueError(f'Invalid value {max_spikes} received for '
                             f'`max_spikes`, expected a non-negative integer value.')
        self.max_spikes = max_spikes
        self.supports_masking = True

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Count the spikes for each neuron
            spike_counts = tf.math.count_nonzero(inputs, axis=0)
            # Create a mask where neurons with spike counts >= max_spikes are set to 0
            mask = tf.cast(tf.less(spike_counts, self.max_spikes))
            return inputs * mask

        output = control_flow_util.smart_cond(training, dropped_inputs, lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'max_spikes': self.max_spikes
        }
        base_config = super(Dropout_custom2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Reduce spike congestion by applying dropout to neurons exceeding a specific latency
class Dropout_custom3(Layer):
    def __init__(self, latency_threshold, **kwargs):
        super(Dropout_custom3, self).__init__(**kwargs)
        if latency_threshold < 0:
            raise ValueError(f'Invalid value {latency_threshold} received for '
                             f'`latency_threshold`, expected a non-negative value.')
        self.latency_threshold = latency_threshold
        self.supports_masking = True

    def call(self, inputs, latency, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Create a binary mask based on latency
            mask = tf.cast(latency <= self.latency_threshold, tf.float32)
            return inputs * mask

        output = control_flow_util.smart_cond(training, dropped_inputs(), lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'latency_threshold': self.latency_threshold
        }
        base_config = super(Dropout_custom3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Send spikes up to max_spikes and ignore excess
class Dropout_custom4(Layer):
    def __init__(self, max_spikes, **kwargs):
        super(Dropout_custom4, self).__init__(**kwargs)
        self.max_spikes = max_spikes
        self.supports_masking = True

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Ensure that the number of spikes does not exceed max_splikes
            return tf.minimum(inputs, self.max_spikes)

        output = tf.cond(training, dropped_inputs, lambda: tf.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'max_spikes': self.max_spikes
        }
        base_config = super(Dropout_custom4, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Apply dropout only to neurons with spikes(greater than 0) among neurons that have passed ReLU
class Dropout_custom6(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout_custom6, self).__init__(**kwargs)
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError(f'Invalid value {rate} received for '
                             f'`rate`, expected a value between 0 and 1.')
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Identify neurons with spikes
            spike_mask = tf.cast(inputs > 0, tf.float32)

            # Apply dropout only to neurons with spikes
            dropout_mask = nn.dropout(
                spike_mask,
                noise_shape=self._get_noise_shape(spike_mask),
                seed=self.seed,
                rate=self.rate)
            return inputs * dropout_mask

        output = control_flow_util.smart_cond(training, dropped_inputs, lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.rate,
            'noise_shape': self.noise_shape,
            'seed': self.seed
        }
        base_config = super(Dropout_custom6, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Apply 0 only to neurons, which have values exceeding specific threshold, after ReLU
class Dropout_custom7(Layer):
    def __init__(self, threshold, **kwargs):
        super(Dropout_custom7, self).__init__(**kwargs)
        if not (0 <= threshold <= 1):
            raise ValueError(f'Invalid value {threshold} received for '
                             f'`threshold`, expected a value between 0 and 1.')
        self.threshold = threshold
        self.supports_masking = True

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Identify neurons with outputs above threshold
            mask = tf.cast(inputs <= self.threshold, tf.float32)
            return inputs * mask

        output = control_flow_util.smart_cond(training, dropped_inputs, lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'threshold': self.threshold
        }
        base_config = super(Dropout_custom7, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Apply dropout only to non-zero inputs
class Dropout_custom9(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout_custom9, self).__init__(**kwargs)
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError(f'Invalid value {rate} received for '
                             f'`rate`, expected a value between 0 and 1.')
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Create a mask for non-zero inputs
            inputs_mask = tf.math.not_equal(inputs, 0)

            # Original dropout mask
            dropout_mask = nn.dropout(
                inputs,
                noise_shape=self._get_noise_shape(inputs),
                seed=self.seed,
                rate=self.rate)

            # Decide whether to apply dropout to non-zero inputs
            combined_mask = tf.cast(inputs_mask, dtype=inputs.dtype) * dropout_mask

            # Calculate with the original inputs to obtain the final output with dropout
            dropout_custom = inputs * combined_mask
            return dropout_custom

        output = control_flow_util.smart_cond(training, dropped_inputs,
                                              lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.rate,
            'noise_shape': self.noise_shape,
            'seed': self.seed
        }
        base_config = super(Dropout_custom9, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Apply dropout only when input tensor element is 1
class Dropout_custom10(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout_custom10, self).__init__(**kwargs)
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError(f'Invalid value {rate} received for '
                             f'`rate`, expected a value between 0 and 1.')
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Create a mask for non-zero locations
            inputs_mask = tf.math.not_equal(inputs, 0)

            # Set dropout targets based on inputs_mask
            dropout_targets = tf.cast(inputs_mask, dtype=inputs.dtype)

            # Create dropout mask only in 1(input tensor element)
            dropout_mask = nn.dropout(
                dropout_targets,
                noise_shape=self._get_noise_shape(inputs),
                seed=self.seed,
                rate=self.rate
            )

            dropout_custom = inputs * dropout_mask
            return dropout_custom

        output = control_flow_util.smart_cond(training, dropped_inputs,
                                              lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.rate,
            'noise_shape': self.noise_shape,
            'seed': self.seed
        }
        base_config = super(Dropout_custom10, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Apply dropout only when input tensor element is 1
class Dropout_custom11(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout_custom11, self).__init__(**kwargs)
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError(f'Invalid value {rate} received for '
                             f'`rate`, expected a value between 0 and 1.')
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Create a mask for locations where inputs are 1
            ones_mask = tf.math.equal(inputs, 1)

            # Get the indices where the inputs are 1
            ones_indices = tf.where(ones_mask)

            # Gather the values from the input tensor at the specified indices
            ones_values = tf.gather_nd(inputs, ones_indices)

            # Apply dropout to these gathered values
            dropped_values = nn.dropout(
                ones_values,
                noise_shape=self._get_noise_shape(ones_values),
                seed=self.seed,
                rate=self.rate)

            # Create a new tensor with the same shape as inputs but with zeros
            zero_tensor = tf.zeros_like(inputs)

            # Update the tensor with the dropped values at the specified indices
            output_tensor = tf.tensor_scatter_nd_update(zero_tensor, ones_indices, dropped_values)
            return output_tensor

        output = control_flow_util.smart_cond(training, dropped_inputs,
                                              lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.rate,
            'noise_shape': self.noise_shape,
            'seed': self.seed
        }
        base_config = super(Dropout_custom11, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Adjust dropout rate based on firing rate and max_d
class Dropout_custom12(Layer):
    def __init__(self, max_d, noise_shape=None, seed=None, **kwargs):
        super(Dropout_custom12, self).__init__(**kwargs)
        if isinstance(max_d, (int, float)) and not 0 <= max_d <= 1:
            raise ValueError(f'Invalid value {max_d} received for '
                             f'`max_d`, expected a value between 0 and 1.')
        self.max_d = max_d
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

    # Calculate dropout rate (Dropout rate varies from 0 to max_d.)
    def compute_d(self, r):
        d = self.max_d * r
        return d

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Calculate firing rate
            r = tf.reduce_mean(inputs)

            # Calculate dropout rate based on r
            d = self.compute_d(r)

            return nn.dropout(
                inputs,
                noise_shape=self._get_noise_shape(inputs),
                seed=self.seed,
                rate=d
            )

        output = control_flow_util.smart_cond(training, dropped_inputs,
                                              lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'max_d': self.max_d,
            'noise_shape': self.noise_shape,
            'seed': self.seed
        }
        base_config = super(Dropout_custom12, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Adjust dropout rate based on firing rate and max_d
class Dropout_custom13(Layer):
    def __init__(self, max_d, noise_shape=None, seed=None, **kwargs):
        super(Dropout_custom13, self).__init__(**kwargs)
        if isinstance(max_d, (int, float)) and not 0 <= max_d <= 1:
            raise ValueError(f'Invalid value {max_d} received for '
                             f'`max_d`, expected a value between 0 and 1.')
        self.max_d = max_d
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Calculate firing rate for each neuron
            r = tf.reduce_mean(inputs, axis=1, keepdims=True)

            # Calculate dropout rate based on r (Dropout rate varies from 0 to max_d.)
            d = self.max_d * r

            # Generate dropout mask with different dropout rates
            random_tensor = tf.random.uniform(shape=tf.shape(inputs), dtype=inputs.type, seed=self.seed)
            keep_mask = random_tensor > d

            return inputs * tf.cast(keep_mask, dtype=inputs.dtype)

        output = control_flow_util.smart_cond(training, dropped_inputs,
                                              lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'max_d': self.max_d,
            'noise_shape': self.noise_shape,
            'seed': self.seed
        }
        base_config = super(Dropout_custom13, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Adjust dropout rate based on firing rate and sigmoid (The minimum value of dropout rate is 0.5.)
# If you want to change the minimum value of dropout rate, change the sigmoid function.
class Dropout_custom14(Layer):
    def __init__(self, noise_shape=None, seed=None, **kwargs):
        super(Dropout_custom14, self).__init__(**kwargs)
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Calculate firing rate for each neuron
            r = tf.reduce_mean(inputs, axis=1, keepdims=True)

            # Calculate dropout rate based on r
            d = r

            # Generate dropout mask with different dropout rates
            random_tensor = tf.random.uniform(shape=tf.shape(inputs), dtype=inputs.type, seed=self.seed)
            keep_mask = random_tensor > d

            return inputs * tf.cast(keep_mask, dtype=inputs.dtype)

        output = control_flow_util.smart_cond(training, dropped_inputs,
                                              lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'noise_shape': self.noise_shape,
            'seed': self.seed
        }
        base_config = super(Dropout_custom14, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Convert 0 to 1
class Dropout_custom15(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout_custom15, self).__init__(**kwargs)
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError(f'Invalid value {rate} received for '
                             f'`rate`, expected a value between 0 and 1.')
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Get a random mask with the same shape as inputs
            random_mask = tf.random.uniform(shape=tf.shape(inputs), minval=0, maxval=1, seed=self.seed)

            # Determine where we should change a 0 to 1
            change_mask = tf.logical_and(tf.math.less(random_mask, self.rate), tf.math.equal(inputs, 0))

            # Use tf.where to get the values from inputs or replace them with 1
            inverted_inputs = tf.where(change_mask, tf.ones_like(inputs), inputs)

            # Compute the scaling factor
            scaling_factor = tf.reduce_sum(inputs) / tf.reduce_sum(inverted_inputs)

            # Scale the inverted_inputs
            return inverted_inputs * scaling_factor

        output = control_flow_util.smart_cond(training, dropped_inputs,
                                              lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.rate,
            'noise_shape': self.noise_shape,
            'seed': self.seed
        }
        base_config = super(Dropout_custom15, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Convert 0 to 1 but do not apply scaling
class Dropout_custom16(Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(Dropout_custom16, self).__init__(**kwargs)
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError(f'Invalid value {rate} received for '
                             f'`rate`, expected a value between 0 and 1.')
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Get a random mask with the same shape as inputs
            random_mask = tf.random.uniform(shape=tf.shape(inputs), minval=0, maxval=1, seed=self.seed)

            # Determine where we should change a 0 to 1
            change_mask = tf.logical_and(tf.math.less(random_mask, self.rate), tf.math.equal(inputs, 0))

            # Use tf.where to get the values from inputs or replace them with 1
            inverted_inputs = tf.where(change_mask, tf.ones_like(inputs), inputs)

            return inverted_inputs

        output = control_flow_util.smart_cond(training, dropped_inputs,
                                              lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.rate,
            'noise_shape': self.noise_shape,
            'seed': self.seed
        }
        base_config = super(Dropout_custom16, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Convert 0 to desired value
class Dropout_custom17(Layer):
    def __init__(self, rate, change_value, noise_shape=None, seed=None, **kwargs):
        super(Dropout_custom17, self).__init__(**kwargs)
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError(f'Invalid value {rate} received for '
                             f'`rate`, expected a value between 0 and 1.')
        self.rate = rate
        self.change_value = change_value
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return None

        concrete_inputs_shape = array_ops.shape(inputs)
        noise_shape = []
        for i, value in enumerate(self.noise_shape):
            noise_shape.append(concrete_inputs_shape[i] if value is None else value)
        return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()

        def dropped_inputs():
            # Get a random mask with the same shape as inputs
            random_mask = tf.random.uniform(shape=tf.shape(inputs), minval=0, maxval=1, seed=self.seed)

            # Determine where we should change a 0 to 1
            change_mask = tf.logical_and(tf.math.less(random_mask, self.rate), tf.math.equal(inputs, 0))

            # Use tf.where to get the values from inputs or replace them with 1
            inverted_inputs = tf.where(change_mask, tf.ones_like(inputs) * self.change_value, inputs)

            # Compute the scaling factor
            scaling_factor = tf.reduce_sum(inputs) / tf.reduce_sum(inverted_inputs)

            # Scale the inverted_inputs
            return inverted_inputs * scaling_factor

        output = control_flow_util.smart_cond(training, dropped_inputs,
                                              lambda: array_ops.identity(inputs))
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'rate': self.rate,
            'change_value': self.change_value,
            'noise_shape': self.noise_shape,
            'seed': self.seed
        }
        base_config = super(Dropout_custom17, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
