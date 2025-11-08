import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import json


class TFLiteOperatorGenerator:
    def __init__(self):
        # 保留简单可靠的算子，删除复杂的算子
        self.supported_ops = [
            'ABS', 'ADD', 'AVERAGE_POOL_2D', 'CAST', 'CEIL',
            'CONCATENATION', 'CONV_2D', 'COS', 'DEPTHWISE_CONV_2D',
            'DIV', 'ELU', 'EQUAL', 'EXP', 'EXPAND_DIMS', 'FLOOR'
            , 'FULLY_CONNECTED', 'GREATER',
            'GREATER_EQUAL', 'HARD_SWISH', 'L2_NORMALIZATION',
            'LEAKY_RELU', 'LESS', 'LESS_EQUAL', 'LOG', 'LOG_SOFTMAX',
            'LOGICAL_AND', 'LOGICAL_NOT', 'LOGICAL_OR', 'LOGISTIC',
            'MAX_POOL_2D', 'MAXIMUM', 'MEAN', 'MINIMUM', 'MUL', 'NEG',
            'NOT_EQUAL', 'RELU', 'RELU6',
            'ROUND', 'RSQRT', 'SIN', 'SOFTMAX', 'SQRT', 'SQUARE',
            'SQUARED_DIFFERENCE', 'SUB', 'SUM', 'TANH'
        ]

        self.size_configs = {
            'small': {'shape_range': [(16, 16), (32, 32)]},
            'medium': {'shape_range': [(32, 32), (64, 64)]},
            'large': {'shape_range': [(64, 64), (128, 128)]}
        }

        # 简化输入维度配置
        self.op_input_dims = {
            'ABS': 4, 'ADD': 4, 'AVERAGE_POOL_2D': 4, 'CAST': 4,
            'CEIL': 4, 'CONCATENATION': 4, 'CONV_2D': 4, 'COS': 4,
            'DEPTHWISE_CONV_2D': 4, 'DIV': 4, 'ELU': 4, 'EQUAL': 4,
            'EXP': 4, 'EXPAND_DIMS': 4, 'FLOOR': 4, 'FLOOR_DIV': 4,
            'FLOOR_MOD': 4, 'FULLY_CONNECTED': 2, 'GREATER': 4,
            'GREATER_EQUAL': 4, 'HARD_SWISH': 4, 'L2_NORMALIZATION': 3,
            'LEAKY_RELU': 4, 'LESS': 4, 'LESS_EQUAL': 4, 'LOG': 4,
            'LOG_SOFTMAX': 2, 'LOGICAL_AND': 4, 'LOGICAL_NOT': 4,
            'LOGICAL_OR': 4, 'LOGISTIC': 4, 'MAX_POOL_2D': 4,
            'MAXIMUM': 4, 'MEAN': 4, 'MINIMUM': 4, 'MUL': 4, 'NEG': 4,
            'NOT_EQUAL': 4, 'PAD': 4, 'PRELU': 4, 'RELU': 4, 'RELU6': 4,
            'RESHAPE': 4, 'ROUND': 4, 'RSQRT': 4, 'SIN': 4, 'SOFTMAX': 4,
            'SQRT': 4, 'SQUARE': 4, 'SQUARED_DIFFERENCE': 4, 'SUB': 4,
            'SUM': 4, 'TANH': 4
        }

        self.op_configs = self._init_op_configs()

    def _init_op_configs(self) -> Dict[str, Dict]:
        """初始化所有算子的参数配置"""
        return {
            'ABS': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'ADD': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'AVERAGE_POOL_2D': {
                'small': {'pool_size': (2, 2), 'strides': (2, 2)},
                'medium': {'pool_size': (3, 3), 'strides': (2, 2)},
                'large': {'pool_size': (4, 4), 'strides': (2, 2)},
                'input_types': [tf.float32]
            },
            'CAST': {
                'small': {'dtype': tf.int32},
                'medium': {'dtype': tf.float16},
                'large': {'dtype': tf.int64},
                'input_types': [tf.float32]
            },
            'CEIL': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'CONCATENATION': {
                'small': {'axis': -1},
                'medium': {'axis': 1},
                'large': {'axis': 2},
                'input_types': [tf.float32, tf.float32]
            },
            'CONV_2D': {
                'small': {'filters': 8, 'kernel_size': (3, 3), 'strides': (1, 1)},
                'medium': {'filters': 16, 'kernel_size': (5, 5), 'strides': (2, 2)},
                'large': {'filters': 32, 'kernel_size': (7, 7), 'strides': (2, 2)},
                'input_types': [tf.float32]
            },
            'COS': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'DEPTHWISE_CONV_2D': {
                'small': {'kernel_size': (3, 3), 'strides': (1, 1), 'depth_multiplier': 1},
                'medium': {'kernel_size': (5, 5), 'strides': (2, 2), 'depth_multiplier': 2},
                'large': {'kernel_size': (7, 7), 'strides': (2, 2), 'depth_multiplier': 4},
                'input_types': [tf.float32]
            },
            'DIV': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'ELU': {
                'small': {'alpha': 1.0}, 'medium': {'alpha': 1.0}, 'large': {'alpha': 1.0},
                'input_types': [tf.float32]
            },
            'EQUAL': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'EXP': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'EXPAND_DIMS': {
                'small': {'axis': -1},
                'medium': {'axis': 1},
                'large': {'axis': 0},
                'input_types': [tf.float32]
            },
            'FLOOR': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'FLOOR_DIV': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.int32, tf.int32]
            },
            'FLOOR_MOD': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.int32, tf.int32]
            },
            'FULLY_CONNECTED': {
                'small': {'units': 64},
                'medium': {'units': 128},
                'large': {'units': 256},
                'input_types': [tf.float32]
            },
            'GREATER': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'GREATER_EQUAL': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'HARD_SWISH': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'L2_NORMALIZATION': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'LEAKY_RELU': {
                'small': {'alpha': 0.1}, 'medium': {'alpha': 0.2}, 'large': {'alpha': 0.3},
                'input_types': [tf.float32]
            },
            'LESS': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'LESS_EQUAL': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'LOG': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'LOG_SOFTMAX': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'LOGICAL_AND': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.bool, tf.bool]
            },
            'LOGICAL_NOT': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.bool]
            },
            'LOGICAL_OR': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.bool, tf.bool]
            },
            'LOGISTIC': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'MAX_POOL_2D': {
                'small': {'pool_size': (2, 2), 'strides': (2, 2)},
                'medium': {'pool_size': (3, 3), 'strides': (2, 2)},
                'large': {'pool_size': (4, 4), 'strides': (2, 2)},
                'input_types': [tf.float32]
            },
            'MAXIMUM': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'MEAN': {
                'small': {'axis': [1, 2], 'keepdims': True},
                'medium': {'axis': [1], 'keepdims': False},
                'large': {'axis': [0, 1], 'keepdims': True},
                'input_types': [tf.float32]
            },
            'MINIMUM': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'MUL': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'NEG': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'NOT_EQUAL': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'PAD': {
                'small': {'padding': [[0, 0], [1, 1], [1, 1], [0, 0]]},
                'medium': {'padding': [[0, 0], [2, 2], [2, 2], [0, 0]]},
                'large': {'padding': [[0, 0], [3, 3], [3, 3], [0, 0]]},
                'input_types': [tf.float32]
            },
            'PRELU': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'RELU': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'RELU6': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'RESHAPE': {
                'small': {'target_shape': [-1, 16, 16, 1]},
                'medium': {'target_shape': [-1, 32, 32, 1]},
                'large': {'target_shape': [-1, 64, 64, 1]},
                'input_types': [tf.float32]
            },
            'ROUND': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'RSQRT': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'SIN': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'SOFTMAX': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'SQRT': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'SQUARE': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            },
            'SQUARED_DIFFERENCE': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'SUB': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32, tf.float32]
            },
            'SUM': {
                'small': {'axis': [1, 2], 'keepdims': True},
                'medium': {'axis': [1], 'keepdims': False},
                'large': {'axis': [0, 1], 'keepdims': True},
                'input_types': [tf.float32]
            },
            'TANH': {
                'small': {}, 'medium': {}, 'large': {},
                'input_types': [tf.float32]
            }
        }

    def get_default_input_shape(self, op_name: str, size: str) -> Tuple:
        """获取算子的默认输入形状"""
        shape_range = self.size_configs[size]['shape_range'][0]

        # 根据算子类型确定正确的输入维度
        dims = self.op_input_dims.get(op_name, 4)  # 默认4维

        if dims == 1:
            return (shape_range[0],)
        elif dims == 2:
            return (shape_range[0], shape_range[1])
        elif dims == 3:
            return (shape_range[0] // 4, shape_range[1] // 4, 8)
        elif dims == 4:
            if op_name in ['CONV_2D', 'DEPTHWISE_CONV_2D', 'MAX_POOL_2D',
                           'AVERAGE_POOL_2D']:
                return (shape_range[0], shape_range[1], 3)
            else:
                return (shape_range[0] // 4, shape_range[1] // 4, 8, 4)

        return (shape_range[0] // 4, shape_range[1] // 4, 8, 4)


def _create_input_layers(input_shape: Tuple, input_types: List) -> List[tf.keras.layers.Input]:
    """创建输入层"""
    inputs = []
    for i, dtype in enumerate(input_types):
        if input_shape is None or len(input_shape) == 0:
            raise ValueError(f"无效的输入形状: {input_shape}")

        inp = tf.keras.layers.Input(shape=input_shape, dtype=dtype, name=f'input_{i}')
        inputs.append(inp)

    return inputs


def _safe_lambda_layer(function, output_shape_fn=None, **kwargs):
    """安全的Lambda层包装器，显式指定输出形状"""
    if output_shape_fn:
        return tf.keras.layers.Lambda(function, output_shape=output_shape_fn, **kwargs)
    else:
        return tf.keras.layers.Lambda(function, **kwargs)


def _build_specific_op(op_name: str, input_shape: Tuple, config: Dict, input_types: List) -> tf.keras.Model:
    """构建具体的算子模型"""

    inputs = _create_input_layers(input_shape, input_types)

    # 为Lambda层定义输出形状函数
    def same_shape(input_shape):
        return input_shape

    def reduce_shape(input_shape, axis=None, keepdims=False):
        if axis is None:
            return (input_shape[0], 1) if keepdims else (input_shape[0],)

        if not isinstance(axis, (list, tuple)):
            axis = [axis]

        new_shape = list(input_shape)
        for ax in axis:
            if ax < 0:
                ax = len(input_shape) + ax
            if 0 <= ax < len(new_shape):
                new_shape[ax] = 1 if keepdims else None

        if not keepdims:
            new_shape = [dim for i, dim in enumerate(new_shape) if i not in axis or dim is not None]

        return tuple([dim for dim in new_shape if dim is not None])

    # 构建具体的算子层
    if op_name == 'ABS':
        x = _safe_lambda_layer(tf.abs, output_shape=same_shape)(inputs[0])
    elif op_name == 'ADD':
        x = tf.keras.layers.Add()(inputs)
    elif op_name == 'AVERAGE_POOL_2D':
        x = tf.keras.layers.AveragePooling2D(
            pool_size=config['pool_size'],
            strides=config['strides'],
            padding='same'
        )(inputs[0])
    elif op_name == 'CAST':
        x = _safe_lambda_layer(
            lambda x: tf.cast(x, config['dtype']),
            output_shape=same_shape
        )(inputs[0])
    elif op_name == 'CEIL':
        x = _safe_lambda_layer(tf.math.ceil, output_shape=same_shape)(inputs[0])
    elif op_name == 'CONCATENATION':
        x = tf.keras.layers.Concatenate(axis=config['axis'])(inputs)
    elif op_name == 'CONV_2D':
        x = tf.keras.layers.Conv2D(
            filters=config['filters'],
            kernel_size=config['kernel_size'],
            strides=config['strides'],
            padding='same'
        )(inputs[0])
    elif op_name == 'COS':
        x = _safe_lambda_layer(tf.cos, output_shape=same_shape)(inputs[0])
    elif op_name == 'DEPTHWISE_CONV_2D':
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=config['kernel_size'],
            strides=config['strides'],
            padding='same',
            depth_multiplier=config['depth_multiplier']
        )(inputs[0])
    elif op_name == 'DIV':
        x = _safe_lambda_layer(
            lambda x: x[0] / x[1],
            output_shape=lambda shapes: shapes[0]
        )(inputs)
    elif op_name == 'ELU':
        x = tf.keras.layers.ELU(alpha=config['alpha'])(inputs[0])
    elif op_name == 'EQUAL':
        x = _safe_lambda_layer(
            lambda x: tf.equal(x[0], x[1]),
            output_shape=lambda shapes: shapes[0]
        )(inputs)
    elif op_name == 'EXP':
        x = _safe_lambda_layer(tf.exp, output_shape=same_shape)(inputs[0])
    elif op_name == 'EXPAND_DIMS':
        # 简化expand_dims实现
        def expand_dims_func(x):
            return tf.expand_dims(x, axis=config['axis'])

        def expand_dims_shape(input_shape):
            new_shape = list(input_shape)
            new_shape.insert(config['axis'] if config['axis'] >= 0 else config['axis'] + len(new_shape) + 1, 1)
            return tuple(new_shape)

        x = _safe_lambda_layer(expand_dims_func, output_shape=expand_dims_shape)(inputs[0])
    elif op_name == 'FLOOR':
        x = _safe_lambda_layer(tf.floor, output_shape=same_shape)(inputs[0])

    # elif op_name == 'FLOOR_MOD':
    #     x = _safe_lambda_layer(
    #         lambda x: tf.floor_mod(x[0], x[1]),
    #         output_shape=lambda shapes: shapes[0]
    #     )(inputs)
    elif op_name == 'FULLY_CONNECTED':
        if len(inputs[0].shape) > 2:
            flattened = tf.keras.layers.Flatten()(inputs[0])
        else:
            flattened = inputs[0]
        x = tf.keras.layers.Dense(units=config['units'])(flattened)
    elif op_name == 'GREATER':
        x = _safe_lambda_layer(
            lambda x: tf.greater(x[0], x[1]),
            output_shape=lambda shapes: shapes[0]
        )(inputs)
    elif op_name == 'GREATER_EQUAL':
        x = _safe_lambda_layer(
            lambda x: tf.greater_equal(x[0], x[1]),
            output_shape=lambda shapes: shapes[0]
        )(inputs)
    elif op_name == 'HARD_SWISH':
        x = _safe_lambda_layer(
            lambda x: x * tf.nn.relu6(x + 3) / 6,
            output_shape=same_shape
        )(inputs[0])
    elif op_name == 'L2_NORMALIZATION':
        x = _safe_lambda_layer(
            lambda x: tf.nn.l2_normalize(x, axis=-1),
            output_shape=same_shape
        )(inputs[0])
    elif op_name == 'LEAKY_RELU':
        x = tf.keras.layers.LeakyReLU(alpha=config['alpha'])(inputs[0])
    elif op_name == 'LESS':
        x = _safe_lambda_layer(
            lambda x: tf.less(x[0], x[1]),
            output_shape=lambda shapes: shapes[0]
        )(inputs)
    elif op_name == 'LESS_EQUAL':
        x = _safe_lambda_layer(
            lambda x: tf.less_equal(x[0], x[1]),
            output_shape=lambda shapes: shapes[0]
        )(inputs)
    elif op_name == 'LOG':
        x = _safe_lambda_layer(tf.math.log, output_shape=same_shape)(inputs[0])
    elif op_name == 'LOG_SOFTMAX':
        x = _safe_lambda_layer(
            lambda x: tf.nn.log_softmax(x),
            output_shape=same_shape
        )(inputs[0])
    elif op_name == 'LOGICAL_AND':
        x = _safe_lambda_layer(
            lambda x: tf.logical_and(x[0], x[1]),
            output_shape=lambda shapes: shapes[0]
        )(inputs)
    elif op_name == 'LOGICAL_NOT':
        x = _safe_lambda_layer(tf.logical_not, output_shape=same_shape)(inputs[0])
    elif op_name == 'LOGICAL_OR':
        x = _safe_lambda_layer(
            lambda x: tf.logical_or(x[0], x[1]),
            output_shape=lambda shapes: shapes[0]
        )(inputs)
    elif op_name == 'LOGISTIC':
        x = tf.keras.layers.Activation('sigmoid')(inputs[0])
    elif op_name == 'MAX_POOL_2D':
        x = tf.keras.layers.MaxPooling2D(
            pool_size=config['pool_size'],
            strides=config['strides'],
            padding='same'
        )(inputs[0])
    elif op_name == 'MAXIMUM':
        x = tf.keras.layers.Maximum()(inputs)
    elif op_name == 'MEAN':
        x = _safe_lambda_layer(
            lambda x: tf.reduce_mean(x, axis=config['axis'], keepdims=config['keepdims']),
            output_shape=lambda shape: reduce_shape(shape, config['axis'], config['keepdims'])
        )(inputs[0])
    elif op_name == 'MINIMUM':
        x = tf.keras.layers.Minimum()(inputs)
    elif op_name == 'MUL':
        x = tf.keras.layers.Multiply()(inputs)
    elif op_name == 'NEG':
        x = _safe_lambda_layer(tf.negative, output_shape=same_shape)(inputs[0])
    elif op_name == 'NOT_EQUAL':
        x = _safe_lambda_layer(
            lambda x: tf.not_equal(x[0], x[1]),
            output_shape=lambda shapes: shapes[0]
        )(inputs)
    # elif op_name == 'PAD':
    #     paddings = tf.constant(config['padding'])
    #     x = _safe_lambda_layer(
    #         lambda x: tf.pad(x, paddings),
    #         output_shape=lambda shape: tuple(shape[i] + sum(config['padding'][i]) for i in range(len(shape)))
    #     )(inputs[0])
    # elif op_name == 'PRELU':
    #     x = tf.keras.layers.PReLU(shared_axes=[1, 2])(inputs[0])
    elif op_name == 'RELU':
        x = tf.keras.layers.ReLU()(inputs[0])
    elif op_name == 'RELU6':
        x = tf.keras.layers.ReLU(max_value=6.0)(inputs[0])
    # elif op_name == 'RESHAPE':
    #     x = tf.keras.layers.Reshape(target_shape=config['target_shape'][1:])(inputs[0])
    elif op_name == 'ROUND':
        x = _safe_lambda_layer(tf.round, output_shape=same_shape)(inputs[0])
    elif op_name == 'RSQRT':
        x = _safe_lambda_layer(
            lambda x: 1.0 / tf.sqrt(x),
            output_shape=same_shape
        )(inputs[0])
    elif op_name == 'SIN':
        x = _safe_lambda_layer(tf.sin, output_shape=same_shape)(inputs[0])
    elif op_name == 'SOFTMAX':
        x = tf.keras.layers.Softmax()(inputs[0])
    elif op_name == 'SQRT':
        x = _safe_lambda_layer(tf.sqrt, output_shape=same_shape)(inputs[0])
    elif op_name == 'SQUARE':
        x = _safe_lambda_layer(tf.square, output_shape=same_shape)(inputs[0])
    elif op_name == 'SQUARED_DIFFERENCE':
        x = _safe_lambda_layer(
            lambda x: tf.square(x[0] - x[1]),
            output_shape=lambda shapes: shapes[0]
        )(inputs)
    elif op_name == 'SUB':
        x = tf.keras.layers.Subtract()(inputs)
    elif op_name == 'SUM':
        x = _safe_lambda_layer(
            lambda x: tf.reduce_sum(x, axis=config['axis'], keepdims=config['keepdims']),
            output_shape=lambda shape: reduce_shape(shape, config['axis'], config['keepdims'])
        )(inputs[0])
    elif op_name == 'TANH':
        x = tf.keras.layers.Activation('tanh')(inputs[0])
    else:
        raise ValueError(f"未实现的算子: {op_name}")

    return tf.keras.Model(inputs=inputs, outputs=x)


def build_op(op_name: str, size: str = 'medium',
             input_shape: Optional[Tuple] = None) -> tf.keras.Model:
    """
    构建单个算子的TensorFlow模型
    """
    generator = TFLiteOperatorGenerator()

    if op_name not in generator.supported_ops:
        raise ValueError(f"不支持的算子: {op_name}")

    if input_shape is None:
        input_shape = generator.get_default_input_shape(op_name, size)

    config = generator.op_configs[op_name][size]
    input_types = generator.op_configs[op_name]['input_types']

    return _build_specific_op(op_name, input_shape, config, input_types)


def convert_to_tflite(tf_model: tf.keras.Model,
                      op_name: str = "unknown") -> bytes:
    """将TensorFlow模型转换为TFLite格式"""

    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.allow_custom_ops = True

    try:
        tflite_model = converter.convert()
        return tflite_model
    except Exception as e:
        print(f"转换失败 {op_name}: {e}")
        return None


def analyze_tflite_model(tflite_model: bytes, op_name: str, size: str):
    """分析TFLite模型结构"""
    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        print(f"\n=== TFLite模型分析: {op_name} ({size}) ===")

        print(interpreter.get_signature_list())
        input_details = interpreter.get_input_details()
        print("输入张量:")
        for i, detail in enumerate(input_details):
            print(f"  Input {i}: shape={detail['shape']}, dtype={detail['dtype']}")

        output_details = interpreter.get_output_details()
        print("输出张量:")
        for i, detail in enumerate(output_details):
            print(f"  Output {i}: shape={detail['shape']}, dtype={detail['dtype']}")

    except Exception as e:
        print(f"分析失败 {op_name}: {e}")


def generate_test_inputs(model: tf.keras.Model, input_types: List) -> List[np.ndarray]:
    """为模型生成测试输入"""
    inputs = []
    for i, input_layer in enumerate(model.inputs):
        shape = [1] + list(input_layer.shape[1:])
        dtype = input_types[i]

        if dtype == tf.float32:
            data = np.random.randn(*shape).astype(np.float32)
        elif dtype == tf.int32:
            data = np.random.randint(0, 10, shape).astype(np.int32)
        elif dtype == tf.int64:
            data = np.random.randint(0, 10, shape).astype(np.int64)
        elif dtype == tf.bool:
            data = np.random.choice([True, False], shape)
        else:
            data = np.random.randn(*shape).astype(np.float32)

        inputs.append(data)

    return inputs


def generate_all_ops(output_dir: str = "./tflite_ops"):
    """生成所有支持的算子模型"""
    generator = TFLiteOperatorGenerator()
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for op_name in generator.supported_ops:
        for size in ['small', 'medium', 'large']:
            try:
                print(f"生成 {op_name} ({size})...")

                model = build_op(op_name, size)
                tflite_model = convert_to_tflite(model, op_name)

                if tflite_model is not None:
                    analyze_tflite_model(tflite_model, op_name, size)

                    filename = f"{op_name}_{size}.tflite"
                    filepath = os.path.join(output_dir, filename)

                    with open(filepath, 'wb') as f:
                        f.write(tflite_model)

                    test_inputs = generate_test_inputs(model, generator.op_configs[op_name]['input_types'])

                    results.append({
                        'op_name': op_name,
                        'size': size,
                        'status': 'success',
                        'filepath': filepath,
                        'input_shapes': [inp.shape for inp in test_inputs]
                    })

                    print(f"✓ 成功生成 {op_name} ({size})")
                else:
                    results.append({
                        'op_name': op_name,
                        'size': size,
                        'status': 'failed',
                        'error': '转换失败'
                    })
                    print(f"✗ 转换失败 {op_name} ({size})")

            except Exception as e:
                results.append({
                    'op_name': op_name,
                    'size': size,
                    'status': 'failed',
                    'error': str(e)
                })
                print(f"✗ 生成失败 {op_name} ({size}): {e}")

    summary = {
        'total_ops': len(generator.supported_ops),
        'success_count': sum(1 for r in results if r['status'] == 'success'),
        'failed_count': sum(1 for r in results if r['status'] == 'failed'),
        'results': results
    }

    with open(os.path.join(output_dir, 'generation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n生成完成! 成功: {summary['success_count']}, 失败: {summary['failed_count']}")
    return summary


def get_op_info(op_name: str) -> Dict:
    """获取算子的详细信息"""
    generator = TFLiteOperatorGenerator()
    if op_name not in generator.supported_ops:
        return None

    info = {
        'name': op_name,
        'supported_sizes': ['small', 'medium', 'large'],
        'input_types': generator.op_configs[op_name]['input_types'],
        'configs': {}
    }

    for size in ['small', 'medium', 'large']:
        info['configs'][size] = generator.op_configs[op_name][size]

    return info


def validate_boolean_op_model(op_name: str, size: str = 'medium', return_errors: bool = False) -> Union[
    bool, Tuple[bool, Dict]]:
    """专门验证布尔输出算子的正确性

    Args:
        op_name: 算子名称
        size: 模型大小
        return_errors: 是否返回误差统计信息

    Returns:
        如果 return_errors=False: 返回布尔值表示验证是否通过
        如果 return_errors=True: 返回元组 (验证结果, 误差统计字典)
    """
    try:
        model = build_op(op_name, size)
        generator = TFLiteOperatorGenerator()
        test_inputs = generate_test_inputs(model, generator.op_configs[op_name]['input_types'])

        # 获取TensorFlow模型预测结果
        tf_output = model.predict(test_inputs)

        # 转换为布尔类型
        tf_bool = tf_output.astype(bool)

        # 转换为TFLite模型
        tflite_model = convert_to_tflite(model, op_name)

        # TFLite推理
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        input_details.reverse()
        for i, (detail, input_data) in enumerate(zip(input_details, test_inputs)):
            interpreter.set_tensor(detail['index'], input_data)

        interpreter.invoke()

        output_details = interpreter.get_output_details()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])

        # 转换为布尔类型
        tflite_bool = tflite_output.astype(bool)

        # 计算布尔输出的统计指标
        total_elements = tf_bool.size

        # 计算匹配和不匹配的数量
        matches = np.sum(tf_bool == tflite_bool)
        mismatches = total_elements - matches

        # 计算错误率
        error_rate = mismatches / total_elements if total_elements > 0 else 0.0
        accuracy = matches / total_elements if total_elements > 0 else 1.0

        # 计算混淆矩阵的各项
        true_positives = np.sum((tf_bool == True) & (tflite_bool == True))
        true_negatives = np.sum((tf_bool == False) & (tflite_bool == False))
        false_positives = np.sum((tf_bool == False) & (tflite_bool == True))
        false_negatives = np.sum((tf_bool == True) & (tflite_bool == False))

        # 计算精确率、召回率和F1分数
        precision = true_positives / (true_positives + false_positives) if (
                                                                                       true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # 验证通过标准：完全匹配（错误率为0）
        is_valid = error_rate == 0.0

        # 构建详细的统计字典
        error_stats = {
            'total_elements': int(total_elements),
            'matches': int(matches),
            'mismatches': int(mismatches),
            'error_rate': float(error_rate),
            'accuracy': float(accuracy),
            'true_positives': int(true_positives),
            'true_negatives': int(true_negatives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'tf_output_true_ratio': float(np.sum(tf_bool) / total_elements) if total_elements > 0 else 0.0,
            'tflite_output_true_ratio': float(np.sum(tflite_bool) / total_elements) if total_elements > 0 else 0.0
        }

        # 输出详细结果
        if is_valid:
            print(f"✓ {op_name} ({size}) 布尔验证通过")
            print(f"  准确率: {accuracy * 100:.2f}%")
            print(f"  错误率: {error_rate * 100:.2f}%")
        else:
            print(f"✗ {op_name} ({size}) 布尔验证失败")
            print(f"  准确率: {accuracy * 100:.2f}%")
            print(f"  错误率: {error_rate * 100:.2f}%")
            print(f"  不匹配数量: {mismatches}/{total_elements}")
            print(f"  真阳性: {true_positives}, 真阴性: {true_negatives}")
            print(f"  假阳性: {false_positives}, 假阴性: {false_negatives}")

            # 输出前几个不匹配的示例
            mismatch_indices = np.where(tf_bool != tflite_bool)[0]
            if len(mismatch_indices) > 0:
                print("  前5个不匹配示例:")
                for i in range(min(5, len(mismatch_indices))):
                    idx = mismatch_indices[i]
                    print(f"    位置 {idx}: TF={tf_bool.flat[idx]}, TFLite={tflite_bool.flat[idx]}")

        if return_errors:
            return is_valid, error_stats
        else:
            return is_valid

    except Exception as e:
        print(f"✗ {op_name} ({size}) 布尔验证失败: {e}")
        import traceback
        traceback.print_exc()

        if return_errors:
            return False, {
                'total_elements': 0,
                'matches': 0,
                'mismatches': 0,
                'error_rate': 1.0,
                'accuracy': 0.0,
                'true_positives': 0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'tf_output_true_ratio': 0.0,
                'tflite_output_true_ratio': 0.0
            }
        else:
            return False

def validate_op_model_enhanced(op_name: str, size: str = 'medium', return_errors: bool = False) -> Union[
    bool, Tuple[bool, Dict]]:
    """增强的验证函数，自动检测输出类型并选择合适的验证方法"""

    try:
        model = build_op(op_name, size)
        generator = TFLiteOperatorGenerator()
        test_inputs = generate_test_inputs(model, generator.op_configs[op_name]['input_types'])

        # 获取TensorFlow模型预测结果
        tf_output = model.predict(test_inputs)

        # 检测输出类型
        output_dtype = tf_output.dtype
        is_boolean_output = (output_dtype == bool or
                             output_dtype == np.bool_ or
                             np.issubdtype(output_dtype, np.bool_))

        # 根据输出类型选择验证方法
        if is_boolean_output:
            return validate_boolean_op_model(op_name, size, return_errors)
        else:
            return validate_numeric_op_model(op_name, size, return_errors)

    except Exception as e:
        print(f"✗ {op_name} ({size}) 验证失败: {e}")
        if return_errors:
            return False, {'error': str(e)}
        else:
            return False


def validate_numeric_op_model(op_name: str, size: str = 'medium', return_errors: bool = False) -> Union[
    bool, Tuple[bool, Dict]]:
    """验证数值输出算子的正确性"""

    try:
        model = build_op(op_name, size)
        generator = TFLiteOperatorGenerator()
        test_inputs = generate_test_inputs(model, generator.op_configs[op_name]['input_types'])

        tf_output = model.predict(test_inputs)

        tflite_model = convert_to_tflite(model, op_name)
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        input_details.reverse()
        for i, (detail, input_data) in enumerate(zip(input_details, test_inputs)):
            interpreter.set_tensor(detail['index'], input_data)

        interpreter.invoke()

        output_details = interpreter.get_output_details()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])

        # 计算各种数值误差指标
        abs_errors = np.abs(tf_output - tflite_output)
        mean_abs_error = np.mean(abs_errors)
        max_abs_error = np.max(abs_errors)

        # 计算相对误差
        relative_errors = np.where(
            np.abs(tf_output) > 1e-10,
            abs_errors / (np.abs(tf_output) + 1e-10),
            0.0
        )
        mean_rel_error = np.mean(relative_errors)
        max_rel_error = np.max(relative_errors)

        # 计算均方根误差
        rmse = np.sqrt(np.mean(np.square(tf_output - tflite_output)))

        # 检查是否通过验证
        is_close = np.allclose(tf_output, tflite_output, atol=1e-5)

        # 构建误差统计字典
        error_stats = {
            'mean_absolute_error': float(mean_abs_error),
            'max_absolute_error': float(max_abs_error),
            'mean_relative_error': float(mean_rel_error),
            'max_relative_error': float(max_rel_error),
            'rmse': float(rmse),
            'tf_output_range': (float(np.min(tf_output)), float(np.max(tf_output))),
            'tflite_output_range': (float(np.min(tflite_output)), float(np.max(tflite_output))),
            'output_dtype': str(tf_output.dtype)
        }

        if is_close:
            print(f"✓ {op_name} ({size}) 数值验证通过")
            print(f"  平均绝对误差: {mean_abs_error:.2e}, 最大绝对误差: {max_abs_error:.2e}")
        else:
            print(f"✗ {op_name} ({size}) 数值验证失败")
            print(f"  平均绝对误差: {mean_abs_error:.2e}, 最大绝对误差: {max_abs_error:.2e}")

        if return_errors:
            return is_close, error_stats
        else:
            return is_close

    except Exception as e:
        print(f"✗ {op_name} ({size}) 数值验证失败: {e}")
        if return_errors:
            return False, {'error': str(e)}
        else:
            return False


# def test_build_op():
#     """测试构建函数"""
#     generator = TFLiteOperatorGenerator()
#
#     for op_name in ['CONV_2D', 'FULLY_CONNECTED', 'ADD', 'MAX_POOL_2D']:
#         for size in ['small', 'medium', 'large']:
#             try:
#                 print(f"测试 {op_name} ({size})...")
#                 model = build_op(op_name, size)
#                 print(f"✓ {op_name} ({size}) 构建成功")
#                 print(f"  输入形状: {model.input_shape}")
#                 print(f"  输出形状: {model.output_shape}")
#
#                 tflite_model = convert_to_tflite(model, op_name)
#                 if tflite_model is not None:
#                     print(f"  TFLite转换成功")
#                     analyze_tflite_model(tflite_model, op_name, size)
#                 else:
#                     print(f"  TFLite转换失败")
#
#             except Exception as e:
#                 print(f"✗ {op_name} ({size}) 失败: {e}")


if __name__ == "__main__":
    # 先测试几个简单的算子
    # test_build_op()

    # 然后生成所有算子
    # summary = generate_all_ops()

    generator = TFLiteOperatorGenerator()

    for op_name  in generator.supported_ops:

        validate_op_model_enhanced(op_name, 'small')