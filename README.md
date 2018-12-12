# Caption to Image with DC-GAN

### Network Architecture
![alt text](https://github.com/andac-demir/Caption2Image-with-DCGAN/blob/master/images/textDCGAN.JPG)

### Parsed Args:
    --inference: if false, train. If true, generate fake images from
                 arbitrary text in the dataset. 

    --dataset: dataset to use. Options: 'flowers' and 'birds'.
    
    --split: an integer that indicates which split to use.
             0: train
             1: validation
             2: test

### Loading Datasets:
  *  We used Caltech-UCSD Birds 200 and Oxford-102 Flowers datasets in training, both are in hd5 format.
  
  * Upload the datasets from the links: [birds](https://drive.google.com/file/d/1mNhn6MYpBb-JwE86GC1kk0VJsYj-Pn5j/view), [flowers](https://drive.google.com/file/d/1EgnaTrlHGaqK5CCgHKLclZMT_AMSTyh8/view) and then move them to your Datasets folder.

### How to run:           
  * first, run the visdom server by: python -m visdom.server

  * then, to train the model do: python main.py 

  * to do inference over the test data: python main.py --inference "true" --split 2

### Network Model
    Based on DC-GAN:
    https://github.com/pytorch/examples/blob/master/dcgan/main.py
    
    First, the text query t is encoded using a fully connected layer to a 
    small dimension (128). 
    Then concatenated to the noise vector z.  
    Following this, concatenated vector is projected to a small spatial extent
    convolutional representation with many feature maps.
    A series of four fractionally-strided convolutions convert this high level
    representation into a 64 x 64 pixel image. 
    No fully connected or pooling layers used.

    Architecture Guidelines:
    - Pooling layers replaced with fractional strided convolutions (generator)
      and strided convolutions (discriminator).
    - Batchnorm used in both the generator and discriminator.
    - Fully connected layers removed for increased depth.
    - In generator ReLU activation used, except for the output which uses Tanh.
    - In discriminator Leaky ReLU activation used for all layers.

