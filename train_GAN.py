import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm_notebook

from configs import BATCH_SIZE, CHANNELS_IMG, device, LEARNING_RATE, checkpoint, Z_DIM, NUM_CLASSES, NUM_ITERATIONS, \
    N_ITER_DISCRI, LAMBDA_GP, E, r, GAMMA
from models import Classifier, Discriminator, Generator
from utils import MyDataSet, load_classifier, load_checkpoint, transform_gan, generate_img, gradient_penalty, gen_loss, \
    save_checkpoint, class_loss

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True # speed up training
    df = pd.read_csv("Dataset/celeba-dataset/list_attr_celeba.csv")
    df = df.replace(-1, 0)
    dataset_gan = MyDataSet(df=df, root_dir="Dataset/celeba-dataset/img_align_celeba/img_align_celeba",
                            transform=transform_gan)
    loader_gan = DataLoader(dataset_gan, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    cla = nn.DataParallel(Classifier(CHANNELS_IMG).to(device))
    try:
        load_classifier(classifier=cla, checkpoint=checkpoint, device=device)
    except:
        print("You need pretrain classifier to train GAN!!!")
        exit()

    gen = nn.DataParallel(Generator(z_dim=Z_DIM, n_classes=NUM_CLASSES).to(device))
    dis = nn.DataParallel(Discriminator(img_channels=CHANNELS_IMG).to(device))
    optim_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE[0], betas=[0, 0.9])
    optim_dis = optim.Adam(dis.parameters(), lr=LEARNING_RATE[0], betas=[0, 0.9])

    try:
        load_checkpoint(gen, dis, optim_gen, optim_dis, checkpoint, device)
    except:
        pass

    cla.eval()
    for p in cla.parameters():
        p.requires_grad = False
    gen_loss_history = []
    dis_loss_history = []
    step = 0
    gamma = GAMMA
    writer_fake = SummaryWriter(f"runs/GAN/Fake")
    writer_scalar = SummaryWriter(f"runs/GAN/Loss")
    one = torch.tensor(1).to(device)
    mone = one * -1
    loaderiter = iter(loader_gan)

    for it in tqdm_notebook(range(NUM_ITERATIONS), desc=f"Training: "):
        if it == 100000:
            for g in optim_dis.param_groups:
                g['lr'] = LEARNING_RATE[1]
            for g in optim_gen.param_groups:
                g['lr'] = LEARNING_RATE[1]
        for p in dis.parameters():
            p.requires_grad = True
        for i in range(N_ITER_DISCRI):
            try:
                datas = next(loaderiter)
            except StopIteration:
                loaderiter = iter(loader_gan)
                datas = next(loaderiter)
            data, labels = datas
            data = data.to(device)  # real images
            batch_size = data.shape[0]
            labels = labels.to(device)
            z = torch.FloatTensor(batch_size, Z_DIM, 1, 1).uniform_(-1, 1).to(device)

            dis.zero_grad()

            # train with real images
            d_loss_real = torch.mean(dis(data))
            d_loss_real.backward(mone)

            # train with fake images
            fake_imgs = gen(z, labels)  # fake images
            d_loss_fake = torch.mean(dis(fake_imgs))
            d_loss_fake.backward(one)

            # train with gradient penalty
            gp = gradient_penalty(dis, data, fake_imgs, device, _ld=LAMBDA_GP)
            gp.backward()

            D_loss = d_loss_fake - d_loss_real + gp

            optim_dis.step()

        for p in dis.parameters():
            p.requires_grad = False

        # Train gen
        gen.zero_grad()
        z = torch.FloatTensor(batch_size, Z_DIM, 1, 1).uniform_(-1, 1).to(device)
        fake_imgs = gen(z, labels)
        output_classifier_fake = cla(fake_imgs)
        output_classifier_real = cla(data)
        output_fake = dis(fake_imgs)
        G_loss = gen_loss(gamma, labels, output_classifier_fake, output_fake)
        G_loss.backward()
        optim_gen.step()

        with torch.no_grad():
            loss_class_fake = class_loss(output_classifier_fake, labels).item()
            loss_class_real = class_loss(output_classifier_real, labels).item()
            gamma = min(20, max(0, gamma + r * (loss_class_fake - E * loss_class_real)))

        if it % 200 == 199:
            with torch.no_grad():
                print(f"G_loss: {round(G_loss.item(), 2)}, D_loss: {round(D_loss.item(), 2)},\
           D_loss_fake: {round(d_loss_fake.item(), 2)}, D_loss_real: {round(d_loss_real.item(), 2)}, gp: {round(gp.item(), 2)}")

                writer_scalar.add_scalar("Loss/Gen", G_loss.item(), global_step=step)
                writer_scalar.add_scalar("Loss/Dis", D_loss.item(), global_step=step)

                fake = generate_img(device, gen, labels, _nums=32)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            save_checkpoint(gen, dis, optim_gen, optim_dis, checkpoint)

            step += 1
