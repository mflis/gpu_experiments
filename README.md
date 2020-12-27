# Reproduction of failures running BERT models on  tensorflow_macos 
This repo provides script that shows failures trying to run `albert-base-v2` model from [transformers](https://huggingface.co/transformers/index.html) package with [tensorflow_macos](https://github.com/apple/tensorflow_macos)

It serves as reproducible case for issue: https://github.com/apple/tensorflow_macos/issues/97

## Error on  tensorflow_macos gpu:

```
/AppleInternal/BuildRoot/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MetalPerformanceShaders-124.0.30/MPSCore/Types/MPSMatrix.mm:241: failed assertion `[MPSMatrix initWithBuffer:descriptor:] buffer may not be nil.'
fish: 'python tf_gpu_experiment.py' terminated by signal SIGABRT (Abort)
```


## Error on  tensorflow_macos gpu:
```
Python(13514,0x70000979d000) malloc: Incorrect checksum for freed object 0x7fd8a3e82800: probably modified after being freed.
Corrupt value: 0x3b7eac9f3bec9097
Python(13514,0x70000979d000) malloc: *** set a breakpoint in malloc_error_break to debug
```

## My hardware:
Macbook Pro 2019
8-core Intel i9, 32 GB RAM, Radeon Pro 555X 4GB
macOS BigSur 11.0.1 (20B50)



## Python packages
Output of `pip list`:
- standard version (tensorflow 2.4.0): packages_tensorflow_2_4_0.txt
- tensorflow_macos version : packages_tensorflow_macos.txt


