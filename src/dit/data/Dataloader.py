from config import data_raw_dir
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class GastroNetDataset(Dataset):
    def __init__(self, img_size=256, transform=None):
        # Initialize dataset, e.g., load data files
        self.data_dir = os.path.join(data_raw_dir, 'GastroNet-spaarne')
        self.data_files = os.listdir(self.data_dir)
        self.transform = transform  # Define any transformations if needed

    def __len__(self):
        # Return the total number of samples
        return len(self.data_files)

    def __getitem__(self, idx):
        # Load and return a sample from the dataset at the given index
        img_path = os.path.join(self.data_dir, self.data_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0
    
def get_gastronet_dataloader(batch_size=32, img_size=256, num_workers=0,):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = GastroNetDataset(img_size=img_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader