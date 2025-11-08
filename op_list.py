import random

tflite_computation_ops = [
    # 基础算术运算
    'ADD',  # 张量加法
    'SUB',  # 张量减法
    'SQUARE',  # 平方运算
    'SQRT',  # 平方根
    'EXP',  # 指数函数
    'LOG',  # 自然对数
    'SIN',  # 正弦函数
    'ABS',  # 绝对值

    # 线性代数运算
    'FULLY_CONNECTED',  # 全连接/矩阵乘法
    'MATMUL',  # 矩阵乘法

    # 卷积运算
    'CONV_2D',  # 2D卷积

    # 池化运算
    'AVERAGE_POOL_2D',  # 平均池化

    # 激活函数
    'RELU',  # ReLU激活
    'TANH',  # 双曲正切
    'SOFTMAX',  # Softmax函数
    'LOG_SOFTMAX',  # Log Softmax
]
tflite_all_op_list = [
    "abs",                     # ABS
    "add",                     # ADD
    "add_n",                   # ADD_N
    "avg_pool2d",           # AVERAGE_POOL_2D
    "matmul",                  # BATCH_MATMUL
    "batch_to_space",       # BATCH_TO_SPACE_ND
    "broadcast_to",            # BROADCAST_TO
    "ceil",                    # CEIL
    "concat",                  # CONCATENATION
    "conv2d",               # CONV_2D
    "cos",                     # COS
    "cumsum",                  # CUMSUM
    "depth_to_space",       # DEPTH_TO_SPACE
    "depthwise_conv2d",     # DEPTHWISE_CONV_2D
    "divide",                  # DIV
    "elu",                  # ELU
    "exp",                     # EXP
    "expand_dims",             # EXPAND_DIMS
    "fill",                    # FILL
    "floor",                   # FLOOR
    "floordiv",                # FLOOR_DIV
    "floormod",                # FLOOR_MOD
    "gather",                  # GATHER
    "gather_nd",               # GATHER_ND
    "hard_swish",           # HARD_SWISH
    "l2_normalize",         # L2_NORMALIZATION
    "leaky_relu",           # LEAKY_RELU
    "log",                     # LOG
    "log_softmax",          # LOG_SOFTMAX
    "math.sigmoid",            # LOGISTIC
    "max_pool2d",           # MAX_POOL_2D
    "maximum",                 # MAXIMUM
    "reduce_mean",             # MEAN
    "minimum",                 # MINIMUM
    "pad",                     # MIRROR_PAD (mode='REFLECT')
    "multiply",                # MUL
    "negative",                # NEG
    "stack",                   # PACK
    "pad",                     # PAD
    "pad",                     # PADV2
    "reduce_max",              # REDUCE_MAX
    "reduce_min",              # REDUCE_MIN
    "relu",                    # RELU
    "relu6",                # RELU6
    "reshape",                 # RESHAPE
    "reverse",                 # REVERSE_V2
    "round",                   # ROUND
    "rsqrt",                   # RSQRT
    "where",                   # SELECT_V2
    "sin",                     # SIN
    "slice",                   # SLICE
    "softmax",              # SOFTMAX
    "space_to_batch",       # SPACE_TO_BATCH_ND
    "space_to_depth",       # SPACE_TO_DEPTH
    "split",                   # SPLIT
    "sqrt",                    # SQRT
    "square",                  # SQUARE
    "squared_difference",      # SQUARED_DIFFERENCE
    "squeeze",                 # SQUEEZE
    "strided_slice",           # STRIDED_SLICE
    "subtract",                # SUB
    "reduce_sum",              # SUM
    "tanh",                    # TANH
    "transpose",               # TRANSPOSE
    "conv2d_transpose",     # TRANSPOSE_CONV
    "unstack",                 # UNPACK
    "zeros_like"               # ZEROS_LIKE
]