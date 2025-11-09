# TFLite与TVM性能对比表

| 算子 | 规模 | 框架 | 常驻内存(MB) | 缺页率(次/秒) | 用户态占比(%) | CPU时间 |
|---|---|---|---|---|---|---|
| ABS_large | 大 | TFLite | 1.6 | 204.891 | 0.3 | 2.8ms |
| ABS_large | 大 | TVM | 2.4 | 625.528 | 0.2 | 31.7ms |
| | | | | | | |
| ABS_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| ABS_medium | 中 | TVM | 2.4 | 638.402 | 0.2 | 31.1ms |
| | | | | | | |
| ABS_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| ABS_small | 小 | TVM | 2.4 | 603.946 | 0.2 | 30.0ms |
| | | | | | | |
| ADD_large | 大 | TFLite | 1.6 | 47.472 | 0.2 | 1.1ms |
| ADD_large | 大 | TVM | 2.5 | 641.093 | 0.2 | 33.9ms |
| | | | | | | |
| ADD_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| ADD_medium | 中 | TVM | 2.5 | 619.498 | 0.2 | 31.1ms |
| | | | | | | |
| ADD_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| ADD_small | 小 | TVM | 2.5 | 609.240 | 0.1 | 30.0ms |
| | | | | | | |
| AVERAGE_POOL_2D_large | 大 | TFLite | 1.6 | 47.321 | 0.2 | 1.1ms |
| AVERAGE_POOL_2D_large | 大 | TVM | 2.4 | 641.098 | 0.2 | 45.6ms |
| | | | | | | |
| AVERAGE_POOL_2D_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| AVERAGE_POOL_2D_medium | 中 | TVM | 2.4 | 621.378 | 0.1 | 35.6ms |
| | | | | | | |
| AVERAGE_POOL_2D_small | 小 | TFLite | 1.5 | 0.000 | 0.0 | 0.0ms |
| AVERAGE_POOL_2D_small | 小 | TVM | 2.5 | 633.105 | 0.2 | 33.3ms |
| | | | | | | |
| CAST_large | 大 | TFLite | 1.6 | 78.376 | 0.1 | 1.1ms |
| CAST_large | 大 | TVM | 2.4 | 632.873 | 0.1 | 31.7ms |
| | | | | | | |
| CAST_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| CAST_medium | 中 | TVM | 2.4 | 593.002 | 0.1 | 30.6ms |
| | | | | | | |
| CAST_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| CAST_small | 小 | TVM | 2.4 | 594.788 | 0.1 | 29.4ms |
| | | | | | | |
| CEIL_large | 大 | TFLite | 1.6 | 104.348 | 0.2 | 1.7ms |
| CEIL_large | 大 | TVM | 2.4 | 665.235 | 0.1 | 32.8ms |
| | | | | | | |
| CEIL_medium | 中 | TFLite | 1.6 | 76.104 | 0.0 | 0.6ms |
| CEIL_medium | 中 | TVM | 2.4 | 592.599 | 0.1 | 29.4ms |
| | | | | | | |
| CEIL_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| CEIL_small | 小 | TVM | 2.4 | 615.858 | 0.2 | 29.4ms |
| | | | | | | |
| CONV_2D_large | 大 | TFLite | 1.9 | 99.123 | 1.0 | 45.6ms |
| CONV_2D_large | 大 | TVM | 2.6 | 741.693 | 0.7 | 166.7ms |
| | | | | | | |
| CONV_2D_medium | 中 | TFLite | 1.6 | 73.734 | 0.4 | 2.8ms |
| CONV_2D_medium | 中 | TVM | 2.5 | 657.452 | 0.2 | 49.4ms |
| | | | | | | |
| CONV_2D_small | 小 | TFLite | 1.6 | 286.919 | 0.1 | 3.3ms |
| CONV_2D_small | 小 | TVM | 3.1 | 641.679 | 0.2 | 36.1ms |
| | | | | | | |
| COS_large | 大 | TFLite | 1.6 | 309.475 | 0.4 | 5.0ms |
| COS_large | 大 | TVM | 3.3 | 642.601 | 0.2 | 33.9ms |
| | | | | | | |
| COS_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| COS_medium | 中 | TVM | 3.1 | 651.838 | 0.3 | 33.9ms |
| | | | | | | |
| COS_small | 小 | TFLite | 1.6 | 103.842 | 0.0 | 0.6ms |
| COS_small | 小 | TVM | 3.1 | 623.825 | 0.1 | 30.6ms |
| | | | | | | |
| DEPTHWISE_CONV_2D_large | 大 | TFLite | 1.8 | 103.345 | 1.0 | 12.8ms |
| DEPTHWISE_CONV_2D_large | 大 | TVM | 2.6 | 99.559 | 0.9 | 16.7ms |
| | | | | | | |
| DEPTHWISE_CONV_2D_medium | 中 | TFLite | 1.6 | 70.898 | 0.2 | 1.1ms |
| DEPTHWISE_CONV_2D_medium | 中 | TVM | 2.5 | 75.444 | 0.3 | 2.2ms |
| | | | | | | |
| DEPTHWISE_CONV_2D_small | 小 | TFLite | 1.6 | 149.361 | 0.1 | 1.1ms |
| DEPTHWISE_CONV_2D_small | 小 | TVM | 3.1 | 116.472 | 0.2 | 2.2ms |
| | | | | | | |
| DIV_large | 大 | TFLite | 1.6 | 142.123 | 0.5 | 3.3ms |
| DIV_large | 大 | TVM | 2.5 | 657.793 | 0.2 | 33.9ms |
| | | | | | | |
| DIV_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| DIV_medium | 中 | TVM | 2.5 | 629.221 | 0.1 | 31.7ms |
| | | | | | | |
| DIV_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| DIV_small | 小 | TVM | 2.5 | 627.696 | 0.2 | 32.2ms |
| | | | | | | |
| ELU_large | 大 | TFLite | 1.6 | 58.870 | 0.1 | 1.1ms |
| ELU_large | 大 | TVM | 3.4 | 629.107 | 0.2 | 37.8ms |
| | | | | | | |
| ELU_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| ELU_medium | 中 | TVM | 3.2 | 622.762 | 0.2 | 31.7ms |
| | | | | | | |
| ELU_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| ELU_small | 小 | TVM | 3.2 | 632.065 | 0.1 | 31.7ms |
| | | | | | | |
| EXPAND_DIMS_large | 大 | TFLite | 1.6 | 39.000 | 0.1 | 0.6ms |
| EXPAND_DIMS_large | 大 | TVM | 2.4 | 70.700 | 0.0 | 1.1ms |
| | | | | | | |
| EXPAND_DIMS_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| EXPAND_DIMS_medium | 中 | TVM | 2.4 | 67.545 | 0.1 | 1.1ms |
| | | | | | | |
| EXPAND_DIMS_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| EXPAND_DIMS_small | 小 | TVM | 2.4 | 0.000 | 0.0 | 0.0ms |
| | | | | | | |
| EXP_large | 大 | TFLite | 1.6 | 65.119 | 0.1 | 1.1ms |
| EXP_large | 大 | TVM | 3.4 | 655.198 | 0.2 | 35.0ms |
| | | | | | | |
| EXP_medium | 中 | TFLite | 1.6 | 74.722 | 0.0 | 0.6ms |
| EXP_medium | 中 | TVM | 3.2 | 633.661 | 0.2 | 32.2ms |
| | | | | | | |
| EXP_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| EXP_small | 小 | TVM | 3.2 | 598.519 | 0.1 | 31.7ms |
| | | | | | | |
| FLOOR_large | 大 | TFLite | 1.6 | 34.000 | 0.0 | 0.6ms |
| FLOOR_large | 大 | TVM | 2.4 | 622.132 | 0.2 | 33.3ms |
| | | | | | | |
| FLOOR_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| FLOOR_medium | 中 | TVM | 2.4 | 613.544 | 0.1 | 31.7ms |
| | | | | | | |
| FLOOR_small | 小 | TFLite | 1.6 | 101.194 | 0.0 | 0.6ms |
| FLOOR_small | 小 | TVM | 2.4 | 608.286 | 0.2 | 30.6ms |
| | | | | | | |
| FULLY_CONNECTED_large | 大 | TFLite | 5.6 | 104.469 | 1.0 | 5.0ms |
| FULLY_CONNECTED_large | 大 | TVM | 15.0 | 688.720 | 0.3 | 96.1ms |
| | | | | | | |
| FULLY_CONNECTED_medium | 中 | TFLite | 2.1 | 88.605 | 0.1 | 1.1ms |
| FULLY_CONNECTED_medium | 中 | TVM | 4.5 | 667.736 | 0.2 | 52.2ms |
| | | | | | | |
| FULLY_CONNECTED_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| FULLY_CONNECTED_small | 小 | TVM | 3.4 | 594.113 | 0.1 | 29.4ms |
| | | | | | | |
| HARD_SWISH_large | 大 | TFLite | 1.6 | 35.658 | 0.1 | 0.6ms |
| HARD_SWISH_large | 大 | TVM | 2.5 | 588.866 | 0.2 | 30.6ms |
| | | | | | | |
| HARD_SWISH_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| HARD_SWISH_medium | 中 | TVM | 2.5 | 638.858 | 0.2 | 32.2ms |
| | | | | | | |
| HARD_SWISH_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| HARD_SWISH_small | 小 | TVM | 2.5 | 622.752 | 0.2 | 32.8ms |
| | | | | | | |
| L2_NORMALIZATION_large | 大 | TFLite | 1.6 | 62.387 | 0.1 | 0.6ms |
| L2_NORMALIZATION_large | 大 | TVM | 2.5 | 650.866 | 0.2 | 35.0ms |
| | | | | | | |
| L2_NORMALIZATION_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| L2_NORMALIZATION_medium | 中 | TVM | 2.5 | 621.779 | 0.1 | 34.4ms |
| | | | | | | |
| L2_NORMALIZATION_small | 小 | TFLite | 1.5 | 0.000 | 0.0 | 0.0ms |
| L2_NORMALIZATION_small | 小 | TVM | 2.5 | 628.807 | 0.2 | 34.4ms |
| | | | | | | |
| LEAKY_RELU_large | 大 | TFLite | 1.6 | 81.699 | 0.1 | 1.1ms |
| LEAKY_RELU_large | 大 | TVM | 2.4 | 641.978 | 0.1 | 32.2ms |
| | | | | | | |
| LEAKY_RELU_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| LEAKY_RELU_medium | 中 | TVM | 2.4 | 619.027 | 0.2 | 33.3ms |
| | | | | | | |
| LEAKY_RELU_small | 小 | TFLite | 1.6 | 77.809 | 0.0 | 0.6ms |
| LEAKY_RELU_small | 小 | TVM | 2.4 | 599.956 | 0.1 | 32.2ms |
| | | | | | | |
| LOGISTIC_large | 大 | TFLite | 1.6 | 207.837 | 0.4 | 3.9ms |
| LOGISTIC_large | 大 | TVM | 3.3 | 657.350 | 0.2 | 39.4ms |
| | | | | | | |
| LOGISTIC_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| LOGISTIC_medium | 中 | TVM | 3.2 | 613.548 | 0.2 | 33.9ms |
| | | | | | | |
| LOGISTIC_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| LOGISTIC_small | 小 | TVM | 3.2 | 642.462 | 0.1 | 33.3ms |
| | | | | | | |
| LOG_SOFTMAX_large | 大 | TFLite | 1.6 | 95.750 | 0.0 | 1.1ms |
| LOG_SOFTMAX_large | 大 | TVM | 3.2 | 647.656 | 0.2 | 33.9ms |
| | | | | | | |
| LOG_SOFTMAX_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| LOG_SOFTMAX_medium | 中 | TVM | 3.2 | 647.559 | 0.1 | 33.3ms |
| | | | | | | |
| LOG_SOFTMAX_small | 小 | TFLite | 1.5 | 99.206 | 0.0 | 0.6ms |
| LOG_SOFTMAX_small | 小 | TVM | 3.2 | 625.424 | 0.2 | 28.3ms |
| | | | | | | |
| LOG_large | 大 | TFLite | 1.6 | 32.413 | 0.1 | 0.6ms |
| LOG_large | 大 | TVM | 3.4 | 669.009 | 0.2 | 37.2ms |
| | | | | | | |
| LOG_medium | 中 | TFLite | 1.6 | 142.601 | 0.1 | 1.1ms |
| LOG_medium | 中 | TVM | 3.2 | 636.283 | 0.2 | 39.4ms |
| | | | | | | |
| LOG_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| LOG_small | 小 | TVM | 3.3 | 628.998 | 0.2 | 33.3ms |
| | | | | | | |
| MAXIMUM_large | 大 | TFLite | 1.6 | 73.383 | 0.2 | 1.7ms |
| MAXIMUM_large | 大 | TVM | 2.5 | 652.845 | 0.3 | 31.1ms |
| | | | | | | |
| MAXIMUM_medium | 中 | TFLite | 1.6 | 58.603 | 0.1 | 0.6ms |
| MAXIMUM_medium | 中 | TVM | 2.5 | 625.650 | 0.2 | 31.7ms |
| | | | | | | |
| MAXIMUM_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| MAXIMUM_small | 小 | TVM | 2.5 | 629.108 | 0.1 | 28.3ms |
| | | | | | | |
| MAX_POOL_2D_large | 大 | TFLite | 1.6 | 68.559 | 0.2 | 1.7ms |
| MAX_POOL_2D_large | 大 | TVM | 2.4 | 637.014 | 0.2 | 32.8ms |
| | | | | | | |
| MAX_POOL_2D_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| MAX_POOL_2D_medium | 中 | TVM | 2.4 | 601.597 | 0.2 | 30.0ms |
| | | | | | | |
| MAX_POOL_2D_small | 小 | TFLite | 1.5 | 0.000 | 0.0 | 0.0ms |
| MAX_POOL_2D_small | 小 | TVM | 2.5 | 616.960 | 0.2 | 32.2ms |
| | | | | | | |
| MEAN_large | 大 | TFLite | 1.6 | 153.760 | 0.2 | 3.3ms |
| MEAN_large | 大 | TVM | 2.4 | 667.415 | 0.2 | 38.3ms |
| | | | | | | |
| MEAN_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| MEAN_medium | 中 | TVM | 2.4 | 626.774 | 0.1 | 35.0ms |
| | | | | | | |
| MEAN_small | 小 | TFLite | 1.5 | 98.503 | 0.0 | 0.6ms |
| MEAN_small | 小 | TVM | 2.4 | 625.520 | 0.1 | 33.3ms |
| | | | | | | |
| MINIMUM_large | 大 | TFLite | 1.6 | 95.642 | 0.2 | 2.2ms |
| MINIMUM_large | 大 | TVM | 2.5 | 650.909 | 0.2 | 33.9ms |
| | | | | | | |
| MINIMUM_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| MINIMUM_medium | 中 | TVM | 2.5 | 641.903 | 0.1 | 32.2ms |
| | | | | | | |
| MINIMUM_small | 小 | TFLite | 1.6 | 82.244 | 0.1 | 0.6ms |
| MINIMUM_small | 小 | TVM | 2.5 | 595.494 | 0.1 | 31.1ms |
| | | | | | | |
| MUL_large | 大 | TFLite | 1.6 | 122.054 | 0.4 | 2.8ms |
| MUL_large | 大 | TVM | 2.5 | 639.016 | 0.1 | 28.9ms |
| | | | | | | |
| MUL_medium | 中 | TFLite | 1.6 | 177.944 | 0.2 | 1.7ms |
| MUL_medium | 中 | TVM | 2.5 | 607.556 | 0.1 | 31.7ms |
| | | | | | | |
| MUL_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| MUL_small | 小 | TVM | 2.5 | 639.722 | 0.1 | 31.1ms |
| | | | | | | |
| NEG_large | 大 | TFLite | 1.6 | 162.931 | 0.1 | 2.2ms |
| NEG_large | 大 | TVM | 2.4 | 624.735 | 0.1 | 29.4ms |
| | | | | | | |
| NEG_medium | 中 | TFLite | 1.6 | 77.754 | 0.1 | 0.6ms |
| NEG_medium | 中 | TVM | 2.4 | 619.105 | 0.1 | 28.3ms |
| | | | | | | |
| NEG_small | 小 | TFLite | 1.6 | 186.489 | 0.0 | 1.1ms |
| NEG_small | 小 | TVM | 2.4 | 588.805 | 0.1 | 30.0ms |
| | | | | | | |
| RELU6_large | 大 | TFLite | 1.6 | 36.719 | 0.1 | 0.6ms |
| RELU6_large | 大 | TVM | 2.4 | 639.592 | 0.2 | 31.1ms |
| | | | | | | |
| RELU6_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| RELU6_medium | 中 | TVM | 2.4 | 606.573 | 0.2 | 28.9ms |
| | | | | | | |
| RELU6_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| RELU6_small | 小 | TVM | 2.4 | 607.194 | 0.0 | 28.9ms |
| | | | | | | |
| RELU_large | 大 | TFLite | 1.6 | 40.955 | 0.0 | 0.6ms |
| RELU_large | 大 | TVM | 2.4 | 612.241 | 0.1 | 31.7ms |
| | | | | | | |
| RELU_medium | 中 | TFLite | 1.6 | 76.051 | 0.1 | 0.6ms |
| RELU_medium | 中 | TVM | 2.4 | 618.075 | 0.1 | 28.9ms |
| | | | | | | |
| RELU_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| RELU_small | 小 | TVM | 2.4 | 627.598 | 0.1 | 27.8ms |
| | | | | | | |
| ROUND_large | 大 | TFLite | 1.6 | 61.380 | 0.0 | 1.1ms |
| ROUND_large | 大 | TVM | 2.4 | 631.282 | 0.2 | 31.1ms |
| | | | | | | |
| ROUND_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| ROUND_medium | 中 | TVM | 2.4 | 601.174 | 0.2 | 30.0ms |
| | | | | | | |
| ROUND_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| ROUND_small | 小 | TVM | 2.4 | 612.705 | 0.1 | 30.6ms |
| | | | | | | |
| RSQRT_large | 大 | TFLite | 1.6 | 161.163 | 0.4 | 2.8ms |
| RSQRT_large | 大 | TVM | 2.4 | 638.347 | 0.2 | 33.3ms |
| | | | | | | |
| RSQRT_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| RSQRT_medium | 中 | TVM | 2.4 | 619.211 | 0.2 | 32.2ms |
| | | | | | | |
| RSQRT_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| RSQRT_small | 小 | TVM | 2.4 | 596.742 | 0.1 | 28.3ms |
| | | | | | | |
| SIN_large | 大 | TFLite | 1.6 | 104.515 | 0.2 | 1.7ms |
| SIN_large | 大 | TVM | 3.3 | 632.194 | 0.1 | 35.6ms |
| | | | | | | |
| SIN_medium | 中 | TFLite | 1.6 | 72.669 | 0.1 | 0.6ms |
| SIN_medium | 中 | TVM | 3.1 | 617.470 | 0.1 | 33.3ms |
| | | | | | | |
| SIN_small | 小 | TFLite | 1.6 | 97.895 | 0.1 | 0.6ms |
| SIN_small | 小 | TVM | 3.1 | 606.346 | 0.1 | 31.1ms |
| | | | | | | |
| SOFTMAX_large | 大 | TFLite | 1.6 | 51.243 | 0.0 | 1.1ms |
| SOFTMAX_large | 大 | TVM | 3.4 | 672.264 | 0.2 | 37.2ms |
| | | | | | | |
| SOFTMAX_medium | 中 | TFLite | 1.6 | 196.466 | 0.2 | 1.7ms |
| SOFTMAX_medium | 中 | TVM | 3.2 | 622.071 | 0.2 | 32.8ms |
| | | | | | | |
| SOFTMAX_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| SOFTMAX_small | 小 | TVM | 3.2 | 607.611 | 0.1 | 30.0ms |
| | | | | | | |
| SQRT_large | 大 | TFLite | 1.6 | 30.813 | 0.1 | 0.6ms |
| SQRT_large | 大 | TVM | 2.4 | 649.647 | 0.2 | 34.4ms |
| | | | | | | |
| SQRT_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| SQRT_medium | 中 | TVM | 2.4 | 621.975 | 0.2 | 30.0ms |
| | | | | | | |
| SQRT_small | 小 | TFLite | 1.6 | 107.875 | 0.0 | 0.6ms |
| SQRT_small | 小 | TVM | 2.4 | 589.820 | 0.1 | 30.6ms |
| | | | | | | |
| SQUARED_DIFFERENCE_large | 大 | TFLite | 1.6 | 95.434 | 0.3 | 2.2ms |
| SQUARED_DIFFERENCE_large | 大 | TVM | 2.5 | 659.097 | 0.2 | 35.6ms |
| | | | | | | |
| SQUARED_DIFFERENCE_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| SQUARED_DIFFERENCE_medium | 中 | TVM | 2.5 | 592.875 | 0.1 | 31.7ms |
| | | | | | | |
| SQUARED_DIFFERENCE_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| SQUARED_DIFFERENCE_small | 小 | TVM | 2.5 | 636.886 | 0.2 | 32.2ms |
| | | | | | | |
| SQUARE_large | 大 | TFLite | 1.6 | 169.783 | 0.1 | 2.2ms |
| SQUARE_large | 大 | TVM | 2.4 | 633.183 | 0.2 | 31.7ms |
| | | | | | | |
| SQUARE_medium | 中 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| SQUARE_medium | 中 | TVM | 2.4 | 597.347 | 0.1 | 28.3ms |
| | | | | | | |
| SQUARE_small | 小 | TFLite | 1.6 | 102.881 | 0.0 | 0.6ms |
| SQUARE_small | 小 | TVM | 2.4 | 635.428 | 0.1 | 31.7ms |
| | | | | | | |
| SUB_large | 大 | TFLite | 1.6 | 143.181 | 0.6 | 3.3ms |
| SUB_large | 大 | TVM | 2.5 | 652.514 | 0.2 | 30.6ms |
| | | | | | | |
| SUB_medium | 中 | TFLite | 1.6 | 172.941 | 0.1 | 1.7ms |
| SUB_medium | 中 | TVM | 2.5 | 601.930 | 0.2 | 35.6ms |
| | | | | | | |
| SUB_small | 小 | TFLite | 1.6 | 0.000 | 0.0 | 0.0ms |
| SUB_small | 小 | TVM | 2.5 | 620.612 | 0.1 | 31.7ms |
| | | | | | | |
| SUM_large | 大 | TFLite | 1.6 | 52.660 | 0.1 | 1.1ms |
| SUM_large | 大 | TVM | 2.4 | 628.740 | 0.1 | 32.2ms |
| | | | | | | |
| SUM_medium | 中 | TFLite | 1.6 | 62.739 | 0.1 | 0.6ms |
| SUM_medium | 中 | TVM | 2.4 | 616.799 | 0.2 | 30.6ms |
| | | | | | | |
| SUM_small | 小 | TFLite | 1.5 | 0.000 | 0.0 | 0.0ms |
| SUM_small | 小 | TVM | 2.4 | 612.419 | 0.2 | 32.2ms |
| | | | | | | |
| TANH_large | 大 | TFLite | 1.6 | 22.872 | 0.1 | 0.6ms |
| TANH_large | 大 | TVM | 3.4 | 666.108 | 0.3 | 41.1ms |
| | | | | | | |
| TANH_medium | 中 | TFLite | 1.6 | 108.994 | 0.1 | 1.1ms |
| TANH_medium | 中 | TVM | 3.2 | 630.719 | 0.2 | 32.8ms |
| | | | | | | |
| TANH_small | 小 | TFLite | 1.6 | 90.481 | 0.0 | 0.6ms |
| TANH_small | 小 | TVM | 3.2 | 658.165 | 0.2 | 30.6ms |
| | | | | | | |
| avg_pool2d | 大 | TFLite | 515.9 | 3.072 | 96.9 | 91.5s |
| avg_pool2d | 大 | TVM | 970.3 | 22.322 | 89.7 | 36.2s |
| | | | | | | |
| avg_pool2d | 中 | TFLite | 26.7 | 2.506 | 97.5 | 1.5s |
| avg_pool2d | 中 | TVM | 29.4 | 128.153 | 51.0 | 649.9ms |
| | | | | | | |
| avg_pool2d | 小 | TFLite | 2.6 | 0.000 | 100.0 | 8.0ms |
| avg_pool2d | 小 | TVM | 2.7 | 79.879 | 50.0 | 8.3ms |
| | | | | | | |
| batch_norm | 大 | TFLite | 678.0 | 8.440 | 91.6 | 44.2s |
| batch_norm | 大 | TVM | 871.2 | 34.729 | 72.0 | 26.1s |
| | | | | | | |
| batch_norm | 中 | TFLite | 33.0 | 4.654 | 95.3 | 1.3s |
| batch_norm | 中 | TVM | 32.7 | 140.660 | 43.2 | 625.8ms |
| | | | | | | |
| batch_norm | 小 | TFLite | 2.5 | 5.294 | 92.0 | 5.2ms |
| batch_norm | 小 | TVM | 2.6 | 59.432 | 54.1 | 6.5ms |
| | | | | | | |
| bias_add | 大 | TFLite | 436.9 | 8.772 | 91.2 | 47.4s |
| bias_add | 大 | TVM | 437.0 | 19.112 | 80.9 | 25.1s |
| | | | | | | |
| bias_add | 中 | TFLite | 22.4 | 3.982 | 96.0 | 1.1s |
| bias_add | 中 | TVM | 22.6 | 8.552 | 91.3 | 531.1ms |
| | | | | | | |
| bias_add | 小 | TFLite | 2.6 | 0.000 | 100.0 | 11.2ms |
| bias_add | 小 | TVM | 2.6 | 11.189 | 81.8 | 7.4ms |
| | | | | | | |
| conv2d | 大 | TFLite | 81.9 | 0.055 | 99.9 | 547.1s |
| conv2d | 大 | TVM | 125.4 | 0.539 | 99.9 | 628.6s |
| | | | | | | |
| conv2d | 中 | TFLite | 7.1 | 0.246 | 99.8 | 3.3s |
| conv2d | 中 | TVM | 10.0 | 7.983 | 98.0 | 2.5s |
| | | | | | | |
| conv2d | 小 | TFLite | 2.6 | 0.000 | 100.0 | 59.9ms |
| conv2d | 小 | TVM | 3.2 | 73.004 | 73.3 | 24.5ms |
| | | | | | | |
| dense | 大 | TFLite | 74.3 | 0.149 | 99.9 | 118.4s |
| dense | 大 | TVM | 114.9 | 10.844 | 97.2 | 23.0s |
| | | | | | | |
| dense | 中 | TFLite | 12.0 | 0.466 | 99.5 | 5.3s |
| dense | 中 | TVM | 21.8 | 33.757 | 90.7 | 1.5s |
| | | | | | | |
| dense | 小 | TFLite | 2.8 | 0.000 | 100.0 | 55.0ms |
| dense | 小 | TVM | 3.4 | 78.087 | 69.8 | 41.1ms |
| | | | | | | |
| matmul | 大 | TFLite | 455.3 | 0.234 | 99.8 | 1191.1s |
| matmul | 大 | TVM | 542.9 | 2.336 | 99.4 | 289.5s |
| | | | | | | |
| matmul | 中 | TFLite | 97.9 | 0.431 | 99.6 | 116.6s |
| matmul | 中 | TVM | 108.2 | 14.390 | 96.1 | 17.5s |
| | | | | | | |
| matmul | 小 | TFLite | 8.4 | 0.597 | 99.4 | 1.4s |
| matmul | 小 | TVM | 9.7 | 77.522 | 76.5 | 331.6ms |
| | | | | | | |
| relu | 大 | TFLite | 678.0 | 8.946 | 91.1 | 41.0s |
| relu | 大 | TVM | 871.2 | 34.806 | 71.8 | 26.0s |
| | | | | | | |
| relu | 中 | TFLite | 678.0 | 8.946 | 91.1 | 41.0s |
| relu | 中 | TVM | 871.2 | 34.806 | 71.8 | 26.0s |
| | | | | | | |
| relu | 小 | TFLite | 21.9 | 8.084 | 91.9 | 1.1s |
| relu | 小 | TVM | 25.4 | 131.007 | 42.9 | 605.6ms |
| | | | | | | |
| softmax | 大 | TFLite | 68.7 | 4.002 | 96.0 | 4.4s |
| softmax | 大 | TVM | 70.3 | 107.965 | 59.5 | 1.9s |
| | | | | | | |
| softmax | 中 | TFLite | 8.0 | 1.704 | 98.3 | 320.0ms |
| softmax | 中 | TVM | 9.8 | 122.320 | 56.6 | 160.1ms |
| | | | | | | |
| softmax | 小 | TFLite | 2.6 | 0.000 | 100.0 | 8.6ms |
| softmax | 小 | TVM | 3.6 | 82.807 | 50.0 | 10.1ms |
| | | | | | | |
