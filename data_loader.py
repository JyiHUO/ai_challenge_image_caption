# -*-coding=utf-8-*-
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
from build_vocab import Vocabulary
import jieba
from torch.autograd import Variable

class myDataset(data.Dataset):
    """Custom Dataset"""
    def __init__(self, root, json, vocab, transform=None):
        '''
        :param root: image dir
        :param json: format json file (ex: [(image_id, caption), (image_id, caption)...])
        :param vocab: vocab wrapper
        :param transform: image transform(import form torchvision)
        '''
        self.root = root
        self.json = json
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """return one data pair
        (image, caption_jieba)
        """
        vocab = self.vocab
        img_name = self.json[index][0]
        img_path = os.path.join(self.root, img_name)
        image = Image.open(img_path).convert("RGB")


        if self.transform is not None:
            image = self.transform(image)

        tokens = jieba.cut(self.json[index][1],cut_all=False)
        caption_jieba = []
        caption_jieba.append(vocab('<start>'))
        caption_jieba.extend([vocab(c) for c in tokens])
        caption_jieba.append(vocab('<end>'))
        return image, torch.Tensor(caption_jieba)

    def __len__(self):
        return len(self.json)

def collate_fn(data):
    """
    Create mini-batch tensors from a list of tuple (image, caption)
    返回的是图片乘以5倍之后的结果

    Args:
        data: 从getitem返回的data类型
            - image: torch tensor of shape
            - caption: [[1,2,3,,,], [4,5,6,,,],,,]
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256)
        targets: torch tensors of shape (batch_size, padded_length)
        lengths: List; valid length for each padded caption
    """
    data.sort(key=lambda x:len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge into 4D
    images = torch.stack(images, 0)

    # Merge captions into 2D
    lengths = [len(cap) for cap in captions]

    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths



def get_loader(root, json, vocab, transform, batch_size, shuffle,num_workers):
    """Returns torch.utils.data.DataLoader for custom dataset"""
    data = myDataset(root=root,
                     json=json,
                     vocab=vocab,
                     transform=transform)
    # 返回 (images, caption, lengths) 对于每一次iteration
    data_loader = torch.utils.data.DataLoader(dataset=data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
