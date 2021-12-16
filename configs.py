import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = [2e-4, 5e-5]
r = .01
ALPHA = .5
BATCH_SIZE = 64
IMAGE_SIZE = 128
CHANNELS_IMG = 3
NUM_CLASSES = 40
Z_DIM = 500
NUM_ITERATIONS = 200000
NUM_EPOCHS_CLF = 20
E = .05
LAMBDA_GP = 10
N_ITER_DISCRI = 5
GAMMA = 0
checkpoint = {
    "gen": "control_gen.pth",
    "disc": "control_disc.pth",
    "class": "control_class.pth",
    "optim_gen": "optim_gen.pth",
    "optim_dis": "optim_dis.pth"
}
