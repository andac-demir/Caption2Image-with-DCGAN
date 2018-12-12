# Caption2Image with DCGAN

### Parsed Args:
    --inference: if false, train. If true, generate fake images from
                 arbitrary text in the dataset. 

    --dataset: dataset to use. Options: 'flowers' and 'birds'.
    
    --split: an integer that indicates which split to use.
             0: train
             1: validation
             2: test

### How to run:           
  * first, run the visdom server by: python -m visdom.server

  * then, to train the model do: python main.py 

  * to do inference: python main.py --inference "true" --split 2

### Network Model
    Based on DC-GAN:
    https://github.com/pytorch/examples/blob/master/dcgan/main.py
    
    Network Architecture:
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

