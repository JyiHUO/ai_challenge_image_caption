import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import json

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def main(args):
    # create model directory
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    with open(args.caption_path, 'rb') as f:
        format_json = json.load(f)

    data_loader = get_loader(args.image_dir, format_json,
                             vocab, transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    # Build the models
    encoder = EncoderCNN(args.embed_size)
    decoder = DecoderRNN(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the Models
    total_step = len(data_loader)

    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):

            print images.size()
            print captions.size()
            # set mini-batch dataset
            images = to_var(images, volatile=True)
            captions = to_var(captions)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            print 'target_size', targets.size()
            # Forward, Backward and Optimize
            decoder.zero_grad()
            encoder.zero_grad()

            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            print 't_output_size', outputs.size()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      % (epoch, args.num_epochs, i, total_step,
                         loss.data[0], np.exp(loss.data[0])))

            # Save the models
            if (i + 1) % args.save_step == 0:
                torch.save(decoder.state_dict(),
                           os.path.join(args.model_path,
                                        'decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
                torch.save(encoder.state_dict(),
                           os.path.join(args.model_path,
                                        'encoder-%d-%d.pkl' % (epoch + 1, i + 1)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='./models/',
                        help="path for saving trained models")
    parser.add_argument('--crop_size', type=int,
                        default=224,
                        help="size for randomly cropping images")
    parser.add_argument('--vocab_path', type=str,
                        default='vocab.pkl',
                        help="path for vocab wrapper")
    parser.add_argument("--image_dir", type=str,
                        default='resize_sub_training_set',
                        help='directory for resized images')
    parser.add_argument('--caption_path',type=str,
                        default='format_caption_train_annotations_20170902.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int,
                        default=10,
                        help="step size for printing log info")
    parser.add_argument("--save_step", type=int, default=100,
                        help='step size for saving trained models')

    # model para
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help="dimension of lstm hidden state")
    parser.add_argument('--num_layers', type=int,default=1,
                        help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=int, default=0.001)

    args = parser.parse_args()
    print (args)
    main(args)