import numpy as np
import os, time, copy
import cv2
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DiceDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.files_loc = [(folder, name) for folder in os.listdir(root_dir) 
                      for name in os.listdir(os.path.join(root_dir, folder))]
        self.transform = transform

    def __len__(self):
        return len(self.files_loc)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
                                self.files_loc[idx][0],
                                self.files_loc[idx][1])
        image = cv2.imread(img_name)
        image = np.swapaxes(image, 0, 2)
        #image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Data Normalization
        # Conversion to float
        image = image.astype(np.float32) 
        
        # Normalization
        image = image/255.0
        
        image = torch.tensor(image)
        
        #sample = {'image': image_grey, 'label': self.files_loc[idx][0]}

        if self.transform:
            sample = self.transform(sample)
        
        return image, int(self.files_loc[idx][0])-1

#def run():
    #print('loop')

#if __name__ == '__main__':
    #run()

if True:
    #dice_types = ['d6']
    
    ## Initialise dataset
    #all_x = np.ones((0, 50, 50), dtype=np.uint8)
    #all_y = np.ones((1), dtype=np.uint8)
    
    ## Import images
    #for dice in dice_types:
        #for face in os.listdir(dice):
            #for image_name in os.listdir(f'{dice}\\{face}'):
                #image = cv2.imread(f'{dice}\\{face}\\{image_name}')
                #image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape((1, 50, 50))
                #all_x = np.append(all_x, image_grey, axis=0)
                #all_y = np.append(all_y, [int(face) - 1])
    
    #N = all_x.shape[0]
    #index = np.arange(N)
    #np.random.shuffle(index)
    
    ## Split data into test and train sets
    #train_N = int(N * 0.8)
    
    #train_images = all_x[index[:train_N]]
    #train_labels = all_y[index[:train_N]]
    
    #val_images = all_x[index[train_N:]]
    #val_labels = all_y[index[train_N:]]
    
    #class_names = np.unique(all_y)
    
    ## Data Normalization
    ## Conversion to float
    #train_images = train_images.astype(np.float32) 
    #test_images = test_images.astype(np.float32)
    
    ## Normalization
    #train_images = train_images/255.0
    #test_images = test_images/255.0
    
    data_dir = 'd6'
    image_datasets = {x: DiceDataset(os.path.join(data_dir, x))
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=0)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = list('123456')
else:
    #data_transforms = {
        #'train': transforms.Compose([
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #]),
        #'val': transforms.Compose([
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            #transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #]),
    #}
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(50),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [255, 255, 255])
        ]),
        'val': transforms.Compose([
            transforms.Resize(50),
            transforms.ToTensor(),
            transforms.Normalize([0, 0, 0], [255, 255, 255])
        ]),
    }
    
    data_dir = 'd6'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=0)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    


#inputs, classes = next(iter(dataloaders['train']))



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                #inputs = torch.tensor(inputs).to(device)
                #labels = torch.tensor(labels).to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = models.mobilenet_v3_small(pretrained=True)
num_ftrs = model_ft.classifier[-1].in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.classifier[-1] = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=1)

torch.save(model_ft, 'tensor.pt')