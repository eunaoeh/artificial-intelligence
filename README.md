# artificial-intelligence

## DnCNN
> Test Image: noisy.png

* Model 1
  * Architecture: in -> conv(3x3x3x64, relu) -> conv(3x3x3x64, relu) -> conv(3x3x3x64, relu) -> conv(3x3x3x64, relu) -> conv(3x3x3x64, relu) -> out
  
* Model 2
  * Model 1 with Skip Connection
  * Architecture: in -> conv(3x3x3x64, relu) -> conv(3x3x3x64, relu) -> conv(3x3x3x64, relu) -> conv(3x3x3x64, relu) -> conv(3x3x3x64, relu) + in -> out
  
* Model 3
  * Model 1 with Skip Connection & Batch Normalization
  * Architecture: in -> conv(3x3x3x64, bn, relu) -> conv(3x3x3x64, bn, relu) -> conv(3x3x3x64, bn, relu) -> conv(3x3x3x64, bn, relu) -> conv(3x3x3x64, bn, relu) + in -> out
