# Vision-Transformer Implementation from scratch
### Pytorch Implementation of ViT Model Presented on [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)
<img src="imgs/ViT-title.PNG" width="500" height="300"></img>


## Vision Transformer - Model Architecture
- Only Encoder from [Transformer](https://arxiv.org/pdf/1706.03762.pdf) is employed. This is not seq2seq model, but an image classifier.
- Classification is performed by one additional token, which is inserted in front of the input token sequence.
<img src="imgs/ViT-architecture.PNG" width="400" height="500"></img>




**Vision Transformer Models Variants**

![image](https://user-images.githubusercontent.com/69974410/185328203-52e36baa-d1a9-4eab-b18d-f92987a71215.png)



**Comparisons with Image Classification Benchmarks**

![image](https://user-images.githubusercontent.com/69974410/185328457-434f53f4-99f5-4161-9fa7-a2945f54faf8.png)


