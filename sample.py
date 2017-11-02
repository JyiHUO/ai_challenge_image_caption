import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import os
from torch.autograd import Variable
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224,224], Image.ANTIALIAS)

    if transform is not None:
        image = transform(image).unsqueeze(0)

    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build Models
    encoder = EncoderCNN(args.embed_size)
    encoder.eval()
    decoder = DecoderRNN(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)

    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare Image
    image = load_image(args.image, transform)
    image_tensor = to_var(image, volatile=True)

    # Use gpu
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Generate caption from image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids.cup().data.numpy()

    # Decode word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word =

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image',type=str,
                        required=True,
                        help='input image for generating caption')
    parser.add_argument('--encoder_path',type=str,
                        default='./models/encoder-1-3300.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path',type=str,
                        default='./models/decoder-1-3300.pkl',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str,
                        default='vocab.pkl',
                        help='path for vocabulary wrapper')

    # Model parameters (should be same as parameters in train.py)
    parser.add_argument('--embed_size',type=int,
                        default=256)
    parser.add_argument('--hidden_size',type=int,
                        default=512)
    parser.add_argument('--num_layers',type=int,
                        default=1)
    args = parser.parse_args()
    main(args)

