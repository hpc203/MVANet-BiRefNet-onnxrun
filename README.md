本套程序对应的文章名称是《Multi-view Aggregation Network for Dichotomous Image Segmentation》，
它是CVPR2024(Highlight)高亮paper，训练源码在https://github.com/qianyu-dlut/MVANet

onnx文件在百度云盘，
链接：https://pan.baidu.com/s/1bd2yJO9F1lYXGQjDKEFCJA 
提取码：rfok


onnx文件大小有430M，这个真有些大了，没法在边缘设备上运行的，而在
Rembg库里使用的通用抠图模型u2netp只有4.7M。


另外一个很火的高精度图像分割模型BiRefNet，它的onnx文件
链接: https://pan.baidu.com/s/1VZ6XyGPUnsO8iqdTwQ9NSw 提取码: 4ah9


它的推理代码跟MVANet的是一样的，因此本套程序也可以用做BiRefNet的推理，只需要修改onnx文件的路径即可。
BiRefNet魔性更大，它的onnx文件有999M
