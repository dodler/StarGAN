import argparse

from data_loader import CustomDataset
from torchvision.transforms import *
from torch.utils.data import DataLoader
from custom_solver import CustomSolver

from PIL import Image

def get_parser():
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--c_dim', type=int, default=5)
    parser.add_argument('--c2_dim', type=int, default=8)
    parser.add_argument('--celebA_crop_size', type=int, default=178)
    parser.add_argument('--rafd_crop_size', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--d_train_repeat', type=int, default=5)

    # Training settings
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['Custom', 'CelebA', 'RaFD', 'Both'])
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--num_iters', type=int, default=200000)
    parser.add_argument('--num_iters_decay', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=str, default=None)

    # Test settings
    parser.add_argument('--test_model', type=str, default='20_1000')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--celebA_image_path', type=str, default='./data/CelebA_nocrop/images')
    parser.add_argument('--utk_image_path', type=str, default='/home/dev/ todo fill me')
    parser.add_argument('--rafd_image_path', type=str, default='./data/RaFD/train')
    parser.add_argument('--metadata_path', type=str, default='./data/list_attr_celeba.txt')
    parser.add_argument('--log_path', type=str, default='./stargan/logs')
    parser.add_argument('--model_save_path', type=str, default='./stargan/models')
    parser.add_argument('--sample_path', type=str, default='./stargan/samples')
    parser.add_argument('--result_path', type=str, default='./stargan/results')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=1000)


crop_size = 178
image_size = 128

path = '/home/dev/Documents/src/recognition/data/UTKFace/aligned_split_cls/'

mode_train = 'train'
mode_val = 'val'

bsize = 16

ds = CustomDataset(path, mode_train)

transform = Compose([(
    CenterCrop((crop_size, crop_size)),
    Resize((image_size, image_size), interpolation=Image.ANTIALIAS),
    RandomHorizontalFlip(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
)])

loader = DataLoader(dataset=ds, batch_size=bsize, shuffle=False)



solver = CustomSolver(loader, get_parser())

solver.train()