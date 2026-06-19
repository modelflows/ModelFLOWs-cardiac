# Model design.

# region Import libraries


import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math

import matplotlib.pyplot as plt

from einops import rearrange
from typing import Optional
from functools import partial


from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables


# endregion


# region ViT for small-size datasets in Keras

# https://keras.io/examples/vision/vit_small_ds/

class ShiftedPatchTokenization(layers.Layer):
    def __init__(
        self,
        image_size=72,
        patch_size=6,
        projection_dim=64,
        vanilla=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vanilla = vanilla  # Flag to switch to vanilla patch extractor
        self.image_size = image_size
        self.patch_size = patch_size
        self.half_patch = patch_size // 2

         # (IMAGE_SIZE // PATCH_SIZE) ** 2
        self.num_patches = (image_size // patch_size) ** 2

        self.flatten_patches = layers.Reshape((self.num_patches, -1))
        self.projection = layers.Dense(units=projection_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def crop_shift_pad(self, images, mode):
        # Build the diagonally shifted images
        if mode == "left-up":
            crop_height = self.half_patch
            crop_width = self.half_patch
            shift_height = 0
            shift_width = 0
        elif mode == "left-down":
            crop_height = 0
            crop_width = self.half_patch
            shift_height = self.half_patch
            shift_width = 0
        elif mode == "right-up":
            crop_height = self.half_patch
            crop_width = 0
            shift_height = 0
            shift_width = self.half_patch
        else:
            crop_height = 0
            crop_width = 0
            shift_height = self.half_patch
            shift_width = self.half_patch

        # Crop the shifted images and pad them
        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch,
        )
        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        return shift_pad

    def call(self, images):
        if not self.vanilla:
            # Concat the shifted images with the original image

            images = tf.concat(
                [
                    images,
                    self.crop_shift_pad(images, mode="left-up"),
                    self.crop_shift_pad(images, mode="left-down"),
                    self.crop_shift_pad(images, mode="right-up"),
                    self.crop_shift_pad(images, mode="right-down"),
                ],
                axis=-1,
            )
        # Patchify the images and flatten it
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        flat_patches = self.flatten_patches(patches)
        if not self.vanilla:
            # Layer normalize the flat patches and linearly project it
            tokens = self.layer_norm(flat_patches)
            tokens = self.projection(tokens)
        else:
            # Linearly project the flat patches
            tokens = self.projection(flat_patches)
        return (tokens, patches)


class PatchEncoder_vit_small_datasets(layers.Layer):
    def __init__(
        self, num_patches=144, projection_dim=64, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)

    def call(self, encoded_patches):
        # print('encoded_patches shape')
        # print(encoded_patches.shape)
        encoded_positions = self.position_embedding(self.positions)
        encoded_patches = encoded_patches + encoded_positions
        return encoded_patches



class MultiHeadAttentionLSA(tf.keras.layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The trainable temperature term. The initial value is
        # the square root of the key dimension.
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0 / self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)

        ### CUSTOM
        # if attention_mask is not None:
          # attention_mask = tf.cast(tf.convert_to_tensor(attention_mask), tf.bool)

        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores


# CHECKED
def mlp_vit_small_datasets(x, hidden_units, dropout_rate, name=''):
    for idx_hidden_unit, units in enumerate(hidden_units):
        x = layers.Dense(units, activation=tf.nn.gelu, name = 'Dense_' + name + '_HiddenUnit_' + str(idx_hidden_unit))(x)
        x = layers.Dropout(dropout_rate, name = 'Dropout_' + name + '_HiddenUnit_' + str(idx_hidden_unit))(x)
    return x
        

def create_vit_classifier(used_data_augmentation_techniques, input_shape = (32, 32, 3), image_size = 72, patch_size=6, transformer_layers = 8, num_heads = 4, projection_dim = 64, transformer_units = [128, 64], mlp_head_units = [2048, 1024], num_classes = 100, vanilla=False, activation = None):


    # Vanilla as 'false' ->  Modified ViT based on the Shifted Patch Tokenization and
    # Locality Self Attention modified ViT

    # Vanilla as 'true' ->  Vanilla ViT
    
    
    num_patches = (image_size // patch_size) ** 2

    inputs = layers.Input(shape=input_shape, name = 'Input_1')
    # Augment data.
    augmented = used_data_augmentation_techniques(inputs)

    used_data_augmentation_techniques.summary()
    # Create patches.
    (tokens, _) = ShiftedPatchTokenization(image_size = image_size, patch_size = patch_size, projection_dim = projection_dim, vanilla=vanilla, name = 'Shifted_Patch_Tokenization_1')(augmented)


    # Encode patches.
    encoded_patches = PatchEncoder_vit_small_datasets(num_patches = num_patches, projection_dim = projection_dim, name = 'Patch_Encoder_1')(tokens)


    # Build the diagonal attention mask
    diag_attn_mask = 1 - tf.eye(num_patches)
    diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)


    # Create multiple layers of the Transformer block.
    for idx_transformer_layer in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6, name = 'Layer_Normalization_1_block_' + str(idx_transformer_layer))(encoded_patches)

        # x1_1 = x1
        # print(x1_1)
        # x1_1._name = 'Layer_Normalization_1_block_' + str(idx_transformer_layer) + '_1'
        # print(x1_1)

        # Create a multi-head attention layer.
        if not vanilla:
            attention_output = MultiHeadAttentionLSA(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1, name = 'MHALSA_block_' + str(idx_transformer_layer),
            )(x1, x1, attention_mask=diag_attn_mask)

        else:
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1, name = 'MHA_block_' + str(idx_transformer_layer),
            )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add(name = 'Add_skip_connection_1_block_' + str(idx_transformer_layer))([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6, name = 'Layer_Normalization_2_block_' + str(idx_transformer_layer))(x2)
        # MLP.
        x3 = mlp_vit_small_datasets(x3, hidden_units=transformer_units, dropout_rate=0.1, name = 'MLP_Block_' + str(idx_transformer_layer))
        # Skip connection 2.
        encoded_patches = layers.Add(name = 'Add_skip_connection_2_block_' + str(idx_transformer_layer))([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6, name = 'Last_Normalization_Layer')(encoded_patches)
    representation = layers.Flatten(name = 'Last_Flatten')(representation)
    representation = layers.Dropout(0.5, name = 'Dropout_prior_MLP')(representation)
    # Add MLP.
    features = mlp_vit_small_datasets(representation, hidden_units=mlp_head_units, dropout_rate=0.5, name = 'Last_MLP')
    # Classify outputs.
    if activation is None:
        logits = layers.Dense(num_classes, name = 'Dense_classifier')(features)
    elif activation == 'softmax':
        logits = layers.Dense(num_classes, name = 'Dense_classifier', activation = 'softmax')(features)

    model = keras.Model(inputs=inputs, outputs=logits, name = 'ViT_Small_Datasets')

    return model


# endregion


# region Customizing functions from tensorflow


def _assert(cond, ex_type, msg):
  """A polymorphic assert, works with tensors and boolean expressions.

  If `cond` is not a tensor, behave like an ordinary assert statement, except
  that a empty list is returned. If `cond` is a tensor, return a list
  containing a single TensorFlow assert op.

  Args:
    cond: Something evaluates to a boolean value. May be a tensor.
    ex_type: The exception class to use.
    msg: The error message.

  Returns:
    A list, containing at most one assert op.
  """
  if _is_tensor(cond):
    return [control_flow_ops.Assert(cond, [msg])]
  else:
    if not cond:
      raise ex_type(msg)
    else:
      return []


def _is_tensor(x):
  """Returns `True` if `x` is a symbolic tensor-like object.

  Args:
    x: A python object to check.

  Returns:
    `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
  """
  return isinstance(x, (ops.Tensor, variables.Variable))



def _CheckAtLeast3DImage(image, require_static=True):
  """Assert that we are working with properly shaped image.

  Args:
    image: >= 3-D Tensor of size [*, height, width, depth]
    require_static: If `True`, requires that all dimensions of `image` are
      known and non-zero.

  Raises:
    ValueError: if image.shape is not a [>= 3] vector.

  Returns:
    An empty list, if `image` has fully defined dimensions. Otherwise, a list
    containing an assert op is returned.
  """
  try:
    if image.get_shape().ndims is None:
      image_shape = image.get_shape().with_rank(3)
    else:
      image_shape = image.get_shape().with_rank_at_least(3)
  except ValueError:
    raise ValueError("'image' must be at least three-dimensional.")
  if require_static and not image_shape.is_fully_defined():
    raise ValueError('\'image\' must be fully defined.')
  if any(x == 0 for x in image_shape):
    raise ValueError('all dims of \'image.shape\' must be > 0: %s' %
                     image_shape)
  if not image_shape.is_fully_defined():
    return [check_ops.assert_positive(array_ops.shape(image),
                                      ["all dims of 'image.shape' "
                                       "must be > 0."])]
  else:
    return []


def _ImageDimensions(image, rank):
  """Returns the dimensions of an image tensor.

  Args:
    image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
    rank: The expected rank of the image

  Returns:
    A list of corresponding to the dimensions of the
    input image.  Dimensions that are statically known are python integers,
    otherwise they are integer scalar tensors.
  """
  if image.get_shape().is_fully_defined():
    return image.get_shape().as_list()
  else:
    static_shape = image.get_shape().with_rank(rank).as_list()
    dynamic_shape = array_ops.unstack(array_ops.shape(image), rank)
    return [s if s is not None else d
            for s, d in zip(static_shape, dynamic_shape)]





def pad_to_bounding_box_custom(image, offset_height, offset_width, target_height,
                                target_width, value_padding = 0):
  """Pad `image` with zeros to the specified `height` and `width`.

  Adds `offset_height` rows of zeros on top, `offset_width` columns of
  zeros on the left, and then pads the image on the bottom and right
  with zeros until it has dimensions `target_height`, `target_width`.

  This op does nothing if `offset_*` is zero and the image already has size
  `target_height` by `target_width`.

  Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or
           3-D Tensor of shape `[height, width, channels]`.
    offset_height: Number of rows of 'value_padding' to add on top.
    offset_width: Number of columns of 'value_padding' to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.
    value_padding: Value of padding.

  Returns:
    If `image` was 4-D, a 4-D float Tensor of shape
    `[batch, target_height, target_width, channels]`
    If `image` was 3-D, a 3-D float Tensor of shape
    `[target_height, target_width, channels]`

  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments, or either `offset_height` or `offset_width` is
      negative.
  """
  image = ops.convert_to_tensor(image, name='image')

  is_batch = True
  image_shape = image.get_shape()
  if image_shape.ndims == 3:
    is_batch = False
    image = array_ops.expand_dims(image, 0)
  elif image_shape.ndims is None:
    is_batch = False
    image = array_ops.expand_dims(image, 0)
    image.set_shape([None] * 4)
  elif image_shape.ndims != 4:
    raise ValueError('\'image\' must have either 3 or 4 dimensions.')

  assert_ops = _CheckAtLeast3DImage(image, require_static=False)

  batch, height, width, depth = _ImageDimensions(image, rank=4)

  after_padding_width = target_width - offset_width - width
  after_padding_height = target_height - offset_height - height

  assert_ops += _assert(offset_height >= 0, ValueError,
                        'offset_height must be >= 0')
  assert_ops += _assert(offset_width >= 0, ValueError,
                        'offset_width must be >= 0')
  assert_ops += _assert(after_padding_width >= 0, ValueError,
                        'width must be <= target - offset')
  assert_ops += _assert(after_padding_height >= 0, ValueError,
                        'height must be <= target - offset')
  image = control_flow_ops.with_dependencies(assert_ops, image)

  # Do not pad on the depth dimensions.
  paddings = array_ops.reshape(
      array_ops.stack([
          0, 0, offset_height, after_padding_height, offset_width,
          after_padding_width, 0, 0
      ]), [4, 2])

  padded = array_ops.pad(image, paddings, constant_values = value_padding)

  padded_shape = [None if _is_tensor(i) else i
                  for i in [batch, target_height, target_width, depth]]
  padded.set_shape(padded_shape)

  if not is_batch:
    padded = array_ops.squeeze(padded, squeeze_dims=[0])

  return padded


# endregion