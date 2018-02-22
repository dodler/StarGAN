from data_loader import CustomDataset
from torchvision.transforms import *
from torch.utils.data import DataLoader

from PIL import Image

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
