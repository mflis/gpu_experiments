
# Reproduction of slow training times using GPU with tensorflow_macos 
This repo provides script that shows slow processing times on macbook GPU compared to CPU using https://github.com/apple/tensorflow_macos
It serves as reproducible case for issue: https://github.com/apple/tensorflow_macos/issues/88

## My hardware:
Macbook Pro 2019
8-core Intel i9, 32 GB RAM, Radeon Pro 555X 4GB

## Python packages
Output of `pip list`:
- standard version (tensorflow 2.4.0): packages_tensorflow_2_4_0.txt
- tensorflow_macos version : packages_tensorflow_macos.txt

## Code changes between experiments
I made few experiments using different batch size and computing device:
To change batch size I modified line:
```
    tfdataset = tf.data.Dataset.from_tensor_slices((features, dataset["labels"])).batch(1)
```
To change computing device I modified line:
```
mlcompute.set_mlc_device(device_name='gpu')
```

## Timings 
Timings were measured after ETA value was stable

```
gpu tensorflow_macos batch 8 -   5/459 [..............................] - ETA: 9:38:35 - loss: 0.8081 - accuracy: 0.4342
gpu tensorflow_macos batch 4 -  5/917 [..............................] - ETA: 4:16:39 - loss: 0.7168 - accuracy: 0.4758
gpu tensorflow_macos batch 1  15/3668 [..............................] - ETA: 3:47:11 - loss: 0.7359 - accuracy: 0.4207

cpu tensorflow_macos batch 1 - 15/3668 [..............................] - ETA: 1:47:14 - loss: 0.7781 - accuracy: 0.4344
cpu tensorflow_macos batch 8  4/459 [..............................] - ETA: 6:54:30 - loss: 0.7636 - accuracy: 0.5104^[

cpu standard tensorflow batch 1 -  23/3668 [..............................] - ETA: 1:31:32 - loss: 0.6736 - accuracy: 0.5025
cpu standard tensorflow batch 8 -   4/459 [..............................] - ETA: 1:31:19 - loss: 0.7148 - accuracy: 0.4635
```



## Logs using standard version of tensorflow

```
2020-12-26 22:22:28.279294: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2020-12-26 22:22:28.279514: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Some layers from the model checkpoint at bert-base-uncased were not used when initializing TFBertForSequenceClassification: ['nsp___cls', 'mlm___cls']
- This IS expected if you are initializing TFBertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing TFBertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some layers of TFBertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier', 'dropout_37']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
2020-12-26 22:23:03.238319: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
```



## Logs using tensorflow_macos
```
WARNING:tensorflow:Eager mode uses the CPU. Switching to the CPU.
2020-12-26 21:52:11.173517: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-12-26 21:52:12.117934: I tensorflow/compiler/tf2mlcompute/utils/mlc_utils.cc:119] Eager mode uses the CPU. Switching to the CPU.
All model checkpoint layers were used when initializing TFBertForSequenceClassification.

Some layers of TFBertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
WARNING:tensorflow:AutoGraph could not transform <bound method TFBertForSequenceClassification.call of <transformers.models.bert.modeling_tf_bert.TFBertForSequenceClassification object at 0x153b6ea90>> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: module 'gast' has no attribute 'Index'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
WARNING:tensorflow:AutoGraph could not transform <bound method TFBertMainLayer.call of <transformers.models.bert.modeling_tf_bert.TFBertMainLayer object at 0x15613c310>> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: module 'gast' has no attribute 'Index'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
WARNING:tensorflow:AutoGraph could not transform <bound method TFBertEmbeddings._embedding of <transformers.models.bert.modeling_tf_bert.TFBertEmbeddings object at 0x15613c130>> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: module 'gast' has no attribute 'Index'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
WARNING:tensorflow:AutoGraph could not transform <bound method TFBertEncoder.call of <transformers.models.bert.modeling_tf_bert.TFBertEncoder object at 0x1561d3ee0>> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: module 'gast' has no attribute 'Index'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
WARNING:tensorflow:AutoGraph could not transform <bound method TFBertLayer.call of <transformers.models.bert.modeling_tf_bert.TFBertLayer object at 0x156263100>> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: module 'gast' has no attribute 'Index'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
WARNING:tensorflow:AutoGraph could not transform <bound method TFBertAttention.call of <transformers.models.bert.modeling_tf_bert.TFBertAttention object at 0x156263250>> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: module 'gast' has no attribute 'Index'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
WARNING:tensorflow:AutoGraph could not transform <bound method TFBertSelfAttention.call of <transformers.models.bert.modeling_tf_bert.TFBertSelfAttention object at 0x1562633a0>> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: module 'gast' has no attribute 'Index'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
WARNING:tensorflow:AutoGraph could not transform <bound method TFBertPooler.call of <transformers.models.bert.modeling_tf_bert.TFBertPooler object at 0x1561d3df0>> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: module 'gast' has no attribute 'Index'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
The parameters `output_attentions`, `output_hidden_states` and `use_cache` cannot be updated when calling a model.They have to be set to True/False in the config object (i.e.: `config=XConfig.from_pretrained('name', output_attentions=True)`).
The parameter `return_dict` cannot be set in graph mode and will always be set to `True`.
2020-12-26 21:52:42.975735: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
```
