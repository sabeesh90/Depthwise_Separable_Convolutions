# EVA6_S7 Assignment

ASSIGNMENT <br>
12 different models were built and executed using various model architectures. The following is the architecture of the models experimented upon. 
The following were the augmentations used in the code <br>
The following were the augmentations used in the code <br>
![augmentations](https://user-images.githubusercontent.com/48343095/123461407-b92fd680-d606-11eb-9876-af4b7511b81f.PNG) <br>

Model 1 – Base model for architecture tuning <br>
Model 2 <br>
1.	There is no dilation block. Total 147,616 parameters
2.	Four Convolutional blocks – 2 layers per block
3.	No separate dilation layer.
4.	3x3 convolution with stride 2 to replicate max pooling like layer. No 1x1 convolution in the max pooling like layer.
5.	Normal sequential passing of layers. No specialized functions such as torch.add to concatenate layer outputs as there is no dilation. 
6.	Highest Accuracy – 84.01 (100 epoch)
7.	Target Accuracy – 84.01 (100epoch)
Model 3 <br>
1.	Total 196,336 parameters
2.	Three convolutional layer per block  - Total Four Convolutional blocks
3.	No separate dilation layer.
4.	3x3 convolution with stride 2 , padding 2 and dilation 2 to replicate (dilation + maxpooling). No 1x1 convolution in max pooling like layer
5.	Normal sequential passing of layers. 
6.	Target Accuracy – 85.51(125 epoch)
7.	Highest Accuracy – 86.32 (236)
Model 4 <br>
1.	This is Similar to model 7 but without the dilation block. Total 147,616 parameters
2.	Four Convolutional blocks – 2 layers per block
3.	No separate dilation layer.
4.	3x3 convolution with stride 2 to replicate max pooling like layer. No 1x1 convolution in the max pooling like layer.
5.	Normal sequential passing of layers. No specialized functions such as torch.add to concatenate layer outputs as there is no dilation. 
6.	Model terminated at epoch 23 as there as no improvement. Highest Accuracy – 75.71 (21 epoch)
7.	Target Accuracy – 75.71 (21 epoch)
Model 5 <br>
1.	This is Similar to model 7 but without the dilation block. Total 147,616 parameters
2.	Four Convolutional blocks - 2 layers per block
3.	No separate dilation layer.
4.	3x3 convolution with stride 2 to replicate max pooling like layer. 1x1 convolution is introduced for the first time in the max pooling like layer.
5.	Normal sequential passing of layers. No specialized functions such as torch.add to concatenate layer outputs as there is no dilation. 
6.	Target Accuracy – 85.19(166epoch)
7.	Highest Accuracy – 85.82(201)
Model 6 <br>
1.	Total 187,296 parameters
2.	Four Convolutional blocks 2 layers per block
3.	No separate dilation layer.
4.	Pure dilation with different kernel sizes (k =10,5,3) in successive  blocks followed  - 1x1 convolution– max pool like layer
5.	Normal sequential passing of layers. No specialized functions such as torch.add to concatenate layer outputs
6.	Highest Accuracy – 77.96 (232 epoch)
7.	Target Accuracy – 77.96 (232 epoch)
Model 7 <br>
1.	Total 153,104 parameters
2.	Four Convolutional blocks 2 layers per block
3.	Dilation layer in third block
4.	No adding of features of dilation layer with normal layer in the third block
5.	3x3 convolution with stride 2 to replicate max pooling like layer.
6.	Target Accuracy – 84.50(248epoch)
7.	Highest Accuracy – 84.50 (248 epoch). Non addition of layers in the dilation block does not result in improvement in performance.
Model 8 <br>
1.	Total 153,104 parameters
2.	Four Convolutional blocks2 layers per block
3.	Dilation layer in third block
4.	Torch. Add layers in the 1st, 2nd and 3rd conv block  - adding of two similar output layers before passing in to max pool like layer
5.	3x3 convolution with stride 2  + 1x1 convolution block – max pool like layer
6.	Target Accuracy – 85.08 (171 epoch)
7.	Highest Accuracy – 85.40 (248 epoch)
Model 9 <br>
1.	Total 197,888 parameters
2.	Four Convolutional blocks
3.	Dilation layer in second convolutional block
4.	Torch. Add layers in the 2nd conv block - adding of two similar output layers before passing in to max pool like layer. 
5.	Pure dilation layer (8,4,2) followed by 1x1 convolution– max pool like layer
6.	There is no significant improvement in model accuracy (Static at 67% validation and 53% training – random model) on using pure dilation layers. Model fails in case of pure dilation layer. 
Model 10 <br> - This is the ideal model
1.	Total 153,104 parameters <br>
![model10 params](https://user-images.githubusercontent.com/48343095/123461785-204d8b00-d607-11eb-8c5b-2b651114d411.PNG) <br>
2.	Four Convolutional blocks
3.	Dilation layer in third block <br>
![dilation](https://user-images.githubusercontent.com/48343095/123462325-cef1cb80-d607-11eb-9631-1e251128aebd.PNG) <br>
4.	Torch. Add layers in the third conv block <br>
![concat](https://user-images.githubusercontent.com/48343095/123462358-dadd8d80-d607-11eb-81c3-c3794ac1a87e.PNG) <br>
5.	3x3 convolution stride 2 followed by 1x1 convolution– max pool like layer <br>
![maxpool](https://user-images.githubusercontent.com/48343095/123462387-e3ce5f00-d607-11eb-8971-e0229f57defb.PNG) <br>
6. four depth wise convolutional layers
![depth2](https://user-images.githubusercontent.com/48343095/123462446-f8aaf280-d607-11eb-9de2-b0ed18ef7b5a.PNG) <br>
![depth3](https://user-images.githubusercontent.com/48343095/123462464-fe083d00-d607-11eb-8d1f-a660412f2002.PNG) <br>
![depth4](https://user-images.githubusercontent.com/48343095/123462480-02ccf100-d608-11eb-9c25-1c6af93d6d90.PNG) <br>
8.	Target Accuracy – 85.09 (139 epoch) <br>
![acc1](https://user-images.githubusercontent.com/48343095/123462870-7a028500-d608-11eb-81e2-e57bc5edaa1e.PNG) <br>
10.	Highest Accuracy – 86.31 (316 epoch) <br>
![acc2](https://user-images.githubusercontent.com/48343095/123462924-85ee4700-d608-11eb-869b-7b91ebebc87c.PNG) <br>
12.	 Receptive field calculation


Model 11 <br>
1.	Total 153,104 parameters
2.	Four Convolutional blocks
3.	Dilation layer in third block
4.	Torch. Add layers in the 1st, 2nd and 3rd conv block - adding of two similar output layers before passing into max pool like layer
5.	3x3 convolution with stride 2 followed by 1x1 convolution– max pool like layer
6.	Target Accuracy – 85.08 (171 epoch)
7.	Highest Accuracy – 85.40 (248 epoch).Accuracy is the same as addition of features of just the dilation block. No contribution of normal layer feature addition.
Model 12 <br>
1.	Total 99,936 parameters
2.	Four convolutions block
3.	Dilation layer in the third block
4.	Torch. Multiplicative layers in the 1st, 2nd and 3rd conv block  - adding of two similar output layers before passing in to max pool like layer
5.	3x3 convolution - followed by 1x1 convolution in stride 2 – max pool like layer
6.	All the layers have depth wise convolution
7.	Target Accuracy – 82.98 (249 epoch)
8.	Highest Accuracy – 82.98 (249 epoch). No significant improvement while using multiplicative features of dilation and non-dilation layers.

Analysis and Findings of the architecture <br>

1.	Reason for normal 3x3 convolution layer following Depth wise convolution layer. A conventional 3x3 convolutional layer has been used in the first layer of every block and in all the layers of the fourth block. It is hypothesized that since depth wise convolution has lesser number of parameters and as initial extraction of features is important in the final prediction, this preliminary feature extraction process cannot be compromised. Lesser parameters means that lesser quality of feature extraction at the initial layers. Adding a normal 3x3 convolution following a depth-wise convolution ensures that there is an increase in parameters and hence the feature learning is not compromised. <br>

2.	Addition of features from layer after the dilated kernel layer. The third convolutional block consists of two layers:- layer without dilation and layer with dilation which extracts same number of feature which same number of output dimension. Due to the dilation of kernel, there is a change in the pattern of feature extraction from the previously trained layers, hence may result in variation of validation accuracy of the model. To prevent this the layers are added using torch.add(). It is hypothesized that this will result in feature augmentation and hence better model performance than without feature addition from layers. <br>

3.	Adding a 1x1 pooling layer after the “max pool like” layer.  Since there is no max pooling layer used here, a kernel of stride 2x2 will result in feature extraction with some features being missed out due to the stride. To compensate for this loss, the feature learning is augmented by using a 1x1 convolution.  As 1x1 convolution sums up the features across channels to result in a new dimensional feature, this property may be exploited to is used as there is no max pooling layer. Hence to prevent loss of features 1x1 is used to add all the features that have been convolved separately  <br>

4.	Torch.add () on normal layers.	It was found that adding the feature output from same channel – same dimension output of two consecutive layers in the same convolutional block did not result in a significant increase in the performance of the model. However, removing Torch.add () from the convolutional block consisting of dilation layers resulted in fall of the performance of the model. This can be hypothesized that the way a feature needs to be extracted is to remain the same (i.e. gradual increase in receptive field) in all the layers. Any sudden increase in the receptive field size results in distortion of the learned features. Hence resulting in drop in performance. Adding the normal output to a dilated output restores this feature learning and results in better model performance. <br>

5.	Torch.mul() on all layers.	Multiplication of features were also experimented upon on all the layers with same dimension – same channel output. It was hypothesized that multiplying the output would result  in more exaggerated feature extraction. But however, this was proved to be incorrect. It is hence hypothesized that multiplying the features from similar output similar dimension channels will result in variation of the extracted features by a multiplicative factor. Hence some features might be over-represented while some may be under-represented. This results in distortion of learning hence reduced model performance. 


