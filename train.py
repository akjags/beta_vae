import time, os, pickle, glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np

from PIL import Image
from model import BetaVAE

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def get_data_loader(rootdir, batch_size=99, imsize=256, norm = {'mean':[0,0,0], 'std':[1,1,1]}):
    train_transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm['mean'], std=norm['std'])#,
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=rootdir, transform=train_transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()

    train_loss = 0

    for batch_idx, images in enumerate(train_loader):
        data, labels = images
        data = data.to(device)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.zero_grad()
        output, mu, logvar = model(data)
        loss = model.loss(output, data, mu, logvar)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()), epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print('Train set Average loss:', train_loss)
    return train_loss

def test(model, device, test_loader, return_images=0, log_interval=None):
    model.eval()
    test_loss = 0
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # two np arrays of images
    original_images = []  
    rect_images = []

    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            data, labels = images
            data = data.to(device)
            output, mu, logvar = model(data)
            loss = model.loss(output, data, mu, logvar)
            test_loss += loss.item()

            if return_images > 0 and len(original_images) < return_images:
                original_images.append(data[0].cpu())
                rect_images.append(output[0].cpu())

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), loss.item()))

    test_loss /= len(test_loader)
    print('Test set Average loss:', test_loss)

    if return_images > 0:
        return test_loss, original_images, rect_images

    return test_loss

def write_log(log_dir, savename, epoch, data, num_to_keep=1):
    """Pickles and writes data to a file
    Args:
        filename(str): File name
        data(pickle-able object): Data to save
    """

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filename = f'{log_dir}/loss_{savename}_e{epoch}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    # Remove old loss files
    loss_files = sorted(glob.glob(f'{log_dir}/*{savename}*'), key=os.path.getmtime)
    for f in loss_files[:-num_to_keep]:
        os.remove(f)


def save_state(save_dir, savename, model, optimizer, epoch, num_to_keep=0):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    file_name = f'{save_dir}/betavae_{savename}_e{epoch}.pt'
    torch.save(checkpoint, file_name)
    print(f'Saved {file_name}\n')
    
    # Delete older save states
    checkpoints = sorted(glob.glob(f'{save_dir}/*{savename}*'), key = os.path.getmtime)
    for ff in checkpoints[:-num_to_keep]:
        print(f'Deleting old checkpoints at: {ff}')
        os.remove(ff)

def load_latest_model(folder, savename, model, optimizer):
    checkpoints = sorted(glob.glob(folder + '/*' + savename + '*'), key=os.path.getmtime)
    if len(checkpoints) == 0:
        print('No prior checkpoints found. Starting training from scratch.')
        return 0
    else:
        print(f'Found a previous checkpoint saved at: {checkpoints[-1]}. Resuming training...')
        return load_state(checkpoints[-1], model, optimizer)+1
    
def load_state(filename, model, optimizer):
    if torch.cuda.is_available():
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location='cpu')

    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return epoch

def load_losses(log_dir, savename):
    loss_files = sorted(glob.glob(log_dir + '/*' + savename + '*'), key=os.path.getmtime)
    if len(loss_files) == 0:
        print('No prior loss files found. Starting training from scratch.')
        return [], []
    else:
        print(f'Found a previous loss file at: {loss_files[-1]}')
        with open(loss_files[-1], 'rb') as f:
            x = pickle.load(f)
        return x[0], x[1]
    
if __name__ == '__main__':
    
    #### SPECIFY DATASET AND VERSION (unique savename)
    DATASET = 'celeba' # either 'celeba' or 'animals'
    ver= 'v2'
    ####################


    base_dir = '/home/users/akshayj/beta_vae'
    SAVE_DIR = f'{base_dir}/checkpoints'
    IMAGE_DIR = f'{base_dir}/images'
    LOG_PATH = f'{base_dir}/losses'
    if DATASET.lower() == 'animals':
        DATADIR = f'{base_dir}/animals'
        IMSIZE=256
    elif DATASET.lower() == 'celeba':
        DATADIR = f'/scratch/groups/jlg/CelebA'
        IMSIZE=64
    else:
        print(f'Model not implemented for this dataset: {DATASET}')
        assert DATASET.lower() in ['animals', 'celeba'], 'dataset not implemented'
        
    LATENT_SIZE = 10 
    BATCH_SIZE = 25
    BETA = 4
    LEARNING_RATE = 1e-5
    LOG_INTERVAL = 100
    EPOCHS = 1000
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: ', device)
    
    # Load data.
    train_loader, test_loader = get_data_loader(DATADIR, batch_size=BATCH_SIZE, imsize=IMSIZE)
    
    # Initialize BetaVAE model with Adam optimizer.
    model = BetaVAE(latent_size=LATENT_SIZE, beta=BETA, imsize=IMSIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-4)
    
    # This will be the unique identifier for checkpoint and loss files.
    SAVENAME=f'{DATASET}_{ver}_beta{BETA}_z{LATENT_SIZE}_{IMSIZE}x{IMSIZE}'
    
    # If we're resuming a previously run model, load the losses and weights and epoch.
    train_losses, test_losses = load_losses(LOG_PATH, SAVENAME)
    start_epoch = load_latest_model(SAVE_DIR, SAVENAME, model, optimizer)
    
    # Begin training and evaluation.
    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, log_interval=LOG_INTERVAL)
        test_loss, original_images, rect_images  = test(model, device, test_loader, return_images=5)
        train_losses.append((epoch, train_loss))
        test_losses.append((epoch, test_loss))

        # Save out images, losses, and model state.
        save_image(original_images + rect_images, f'{IMAGE_DIR}/images_{SAVENAME}_e{epoch}.png', 
                   padding=0, nrow=len(original_images))
        write_log(LOG_PATH, SAVENAME, epoch, (train_losses, test_losses))
        save_state(SAVE_DIR, SAVENAME, model, optimizer, epoch, num_to_keep=5)
            
