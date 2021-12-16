import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm_notebook

from configs import BATCH_SIZE, CHANNELS_IMG, device, LEARNING_RATE, checkpoint, NUM_EPOCHS_CLF
from models import Classifier
from utils import MyDataSet, transform, load_classifier, class_loss, save_classifier

if __name__ == "__main__":
    df = pd.read_csv("Dataset/celeba-dataset/list_attr_celeba.csv")
    df = df.replace(-1, 0)
    dataset = MyDataSet(df=df, root_dir="Dataset/celeba-dataset/img_align_celeba/img_align_celeba",
                        transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    cla = nn.DataParallel(Classifier(CHANNELS_IMG).to(device))
    optim_cla = optim.Adam(cla.parameters(), lr=LEARNING_RATE[0], betas=(0.9, 0.999))

    try:
        load_classifier(classifier=cla, checkpoint=checkpoint, device=device)
    except:
        pass

    """
    + Pretrain classifier with 20 epochs to help GAN can generate controllable image.
    + We dont need to evaluation this model, because we only use this for classifying the image in CelebA dataset,
    so overfitting is acceptable. Just train for loss_train min as possible, this is all thing we need here <3
    """

    cla.train()
    for epoch in range(NUM_EPOCHS_CLF):
        for index, (data, labels) in enumerate(tqdm_notebook(loader, desc=f"Epoch: {epoch}")):
            data = data.to(device)
            labels = labels.to(device)
            cla.zero_grad()
            output = cla(data)
            loss = class_loss(output, labels)
            loss.backward()
            optim_cla.step()
            if index % 500 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, index + 1, loss))
                save_classifier(cla, checkpoint=checkpoint)
