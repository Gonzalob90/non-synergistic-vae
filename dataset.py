import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import zipfile

DSPRITES_PATH = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
CELEBA_PATH = 'img_align_celeba.zip'
NUM_EXAMPLES_CELEBA = 202599
#NUM_EXAMPLES_CELEBA = 5000

class SpritesDataset(Dataset):
    def __init__(self):
        self.dataset_zip = np.load(DSPRITES_PATH, encoding='latin1')
        self.imgs = self.dataset_zip['imgs']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x1 = self.imgs[idx].astype(np.float32)
        idx2 = np.random.randint(len(self))
        x2 = self.imgs[idx2].astype(np.float32)
        return x1, x2


def get_dsprites_dataloader(batch_size, shuffle=True):
    sprite_dataset = SpritesDataset()
    return DataLoader(sprite_dataset, batch_size=batch_size, shuffle=shuffle)



def convert_celeba_64(image_file_path):

    dataset_clean = []

    with zipfile.ZipFile(image_file_path, 'r') as image_file:

            for i in range(NUM_EXAMPLES_CELEBA):

                image_name = 'img_align_celeba/{:06d}.jpg'.format(i + 1)
                image = Image.open(
                    image_file.open(image_name, 'r')).resize(
                        (64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
                dataset_clean.append(np.asarray(image).transpose(2, 0, 1) / 255)

    dataset_clean = np.array(dataset_clean)
    #print(dataset_clean.shape)
    return dataset_clean


def convert_celeba_64_a():

    dataset_clean = []

    for i in range(NUM_EXAMPLES_CELEBA):

        image_name = 'img_align_celeba/{:06d}.jpg'.format(i + 1)
        image = Image.open(image_name, 'r').resize((64, 78), Image.ANTIALIAS).crop((0, 7, 64, 64 + 7))
        dataset_clean.append(np.asarray(image).transpose(2, 0, 1) / 255)

    dataset_clean = np.array(dataset_clean)
    #print(dataset_clean.shape)
    return dataset_clean


class CelebADataset(Dataset):
    def __init__(self):
        self.imgs = convert_celeba_64_a()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x1 = self.imgs[idx].astype(np.float32)
        idx2 = np.random.randint(len(self))
        x2 = self.imgs[idx2].astype(np.float32)
        return x1, x2

class CelebADataset_GPU(Dataset):
    def __init__(self):
        self.imgs = convert_celeba_64()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x1 = self.imgs[idx].astype(np.float32)
        idx2 = np.random.randint(len(self))
        x2 = self.imgs[idx2].astype(np.float32)
        return x1, x2

def get_celeba_dataloader(batch_size, shuffle=True):
    celeba_dataset = CelebADataset()
    return DataLoader(celeba_dataset, batch_size=batch_size, shuffle=shuffle)

def get_celeba_dataloader_gpu(batch_size, shuffle=True):
    celeba_dataset = CelebADataset_GPU()
    return DataLoader(celeba_dataset, batch_size=batch_size, shuffle=shuffle)

