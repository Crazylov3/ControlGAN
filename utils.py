import os

import PIL.Image as Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tf
from torch.utils.data import Dataset

from configs import Z_DIM


class MyDataSet(Dataset):
    def __init__(self, df, root_dir, transform=None):
        super(MyDataSet, self).__init__()
        self.annotations = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        rootdir = self.root_dir + f"/{index // 10000}"
        img_path = os.path.join(rootdir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert("RGB")
        y_label = torch.from_numpy(np.array(self.annotations.iloc[index, 1:], dtype="float32"))
        if self.transform:
            image = self.transform(image)
        return image, y_label


def gradient_penalty(dis, real, fake, device, _ld=10):
    b, c, h, w = real.shape
    alpha = torch.rand((b, 1, 1, 1)).repeat(1, c, h, w).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)
    interpolated_images = torch.autograd.Variable(interpolated_images, requires_grad=True)
    mixed_scores = dis(interpolated_images)

    gradient = torch.autograd.grad(inputs=interpolated_images,
                                   outputs=mixed_scores,
                                   grad_outputs=torch.ones_like(mixed_scores),
                                   create_graph=True,
                                   retain_graph=True, )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty * _ld


def gen_loss(gamma, labels, classifier_output, discriminator_output):  # generator loss
    return gamma * class_loss(classifier_output, labels) - torch.mean(discriminator_output)


def class_loss(input, labels):  # classifier loss
    loss = nn.BCELoss()
    return loss(input, labels)


def save_classifier(classifier, checkpoint):
    torch.save(classifier.state_dict(), checkpoint['class'])
    print("=====> Saved checkpoint")

def load_classifier(classifier, checkpoint, device):
    classifier.load_state_dict(torch.load(checkpoint['class'], map_location=device))
    print("=====> Loaded checkpoint")

def transform(img):
    return tf.Compose(
        [
            tf.Resize((128, 128)),
            tf.ColorJitter(brightness=0.5),
            tf.RandomHorizontalFlip(p=0.5),
            tf.ToTensor(),
            tf.Normalize((0.5,), (0.5,))
        ]
    )(img)


def transform_gan(img):
    return tf.Compose(
        [
            tf.Resize((128, 128)),
            tf.ToTensor(),
            tf.Normalize((0.5,), (0.5,))
        ]
    )(img)


def load_checkpoint(gen, disc, optim_gen, optim_dis, checkpoint, device):
    gen.load_state_dict(torch.load(checkpoint["gen"], map_location=device))
    disc.load_state_dict(torch.load(checkpoint["disc"], map_location=device))
    optim_gen.load_state_dict(torch.load(checkpoint["optim_gen"]))
    optim_dis.load_state_dict(torch.load(checkpoint["optim_dis"]))
    print("=====> Loaded checkpoint")


def save_checkpoint(gen, disc, optim_gen, optim_dis, checkpoint):
    torch.save(gen.state_dict(), checkpoint["gen"])
    torch.save(disc.state_dict(), checkpoint["disc"])
    torch.save(optim_gen.state_dict(), checkpoint["optim_gen"])
    torch.save(optim_dis.state_dict(), checkpoint["optim_dis"])
    print("=====> Saved checkpoint")


def generate_img(device, netG, labels, _nums=32):
    nums = min(32, labels.shape[0])
    fixed_noise = torch.FloatTensor(nums, Z_DIM, 1, 1).uniform_(-1, 1).to(device)
    samples = netG(fixed_noise, labels[:nums])
    samples = samples.mul(0.5).add(0.5)
    return samples
