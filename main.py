'''
    inference: if false, train. If true, generate fake images from
               arbitrary text in the dataset. 
    dataset: dataset to use. Options: 'flowers' and 'birds'.
    split: an integer that indicates which split to use.
           0: train
           1: validation
           2: test
'''
from train import Train
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", type=str2bool, default='f')
    parser.add_argument('--dataset', type=str, default='flowers')
    parser.add_argument('--split', default=0, type=int)
    parser.add_argument('--optimization', type=str, default='lbfgs')
    args = parser.parse_args()
    return args

def main():
    args = parseArgs()
    train = Train(dataset=args.dataset.lower(),
                  split=args.split,
                  lr=0.0002,
                  l1_coef=50,
                  l2_coef=100,
                  batch_size=128,
                  num_workers=8,
                  epochs=120,
                  optimization=args.optimization.lower())
    if not args.inference:
        train.train_network()
    else:
        train.predict()


if __name__ == "__main__":
    main()

