from torch.utils.data import Dataset, DataLoader
import random
from my_dataset_coco import CocoDetection
from transforms import Compose, ToTensor
from utils import *

class TripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = [label['labels'][0].item() if len(label['labels']) > 0 else -1 for _, label in dataset]

    def __getitem__(self, index):
        anchor_img, anchor_target = self.dataset[index]
        anchor_label = anchor_target['labels'][0].item() if len(anchor_target['labels']) > 0 else -1

        positive_index = index
        while positive_index == index or self.labels[positive_index] != anchor_label:
            positive_index = random.choice(range(len(self.dataset)))
        positive_img, _ = self.dataset[positive_index]

        negative_index = random.choice([i for i in range(len(self.dataset)) if self.labels[i] != anchor_label])
        negative_img, _ = self.dataset[negative_index]

        return anchor_img, positive_img, negative_img

    def __len__(self):
        return len(self.dataset)

class Decoder(nn.Module):
    def __init__(self, feature_dim, out_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(feature_dim, 512 * 4 * 4)
        self.upsample1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.upsample2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.upsample3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.final_deconv = nn.ConvTranspose2d(64, out_channels, 4, 2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        x = self.relu(self.upsample1(x))
        x = self.relu(self.upsample2(x))
        x = self.relu(self.upsample3(x))
        x = self.final_deconv(x)
        x = self.sigmoid(x)
        return x

class AutoEncoderWithAttention(nn.Module):
    def __init__(self, in_channels, feature_dim, out_channels):
        super(AutoEncoderWithAttention, self).__init__()
        self.encoder = EncoderWithAttention(in_channels, feature_dim)
        self.decoder = Decoder(feature_dim, out_channels)

    def forward(self, x):
        encoded_features = self.encoder(x)
        reconstructed_image = self.decoder(encoded_features)
        return reconstructed_image


if __name__ == '__main__':
    transforms = Compose([ToTensor()])
    data_root = './archive'
    train_dataset = CocoDetection(root=data_root, dataset="train", transforms=transforms)
    test_dataset = CocoDetection(root=data_root, dataset="val", transforms=transforms)
    train_triplet_dataset = TripletDataset(train_dataset)
    train_triplet_loader = DataLoader(train_triplet_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_triplet_dataset = TripletDataset(test_dataset)
    val_triplet_loader = DataLoader(val_triplet_dataset, batch_size=8, shuffle=True, num_workers=8)

    input_dim = 640
    hidden_dim = 1024
    feature_dim = 512
    output_dim = input_dim
    margin = 5.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoEncoderWithAttention(in_channels=3, feature_dim=256, out_channels=3)
    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss_fn = EnhancedContrastiveLoss(margin=5.0, alpha=1.0, beta=1.0)

    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_triplet_loader:
            anchor, positive, negative = batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()

            anchor_encoded = model.encoder(anchor)
            positive_encoded = model.encoder(positive)
            negative_encoded = model.encoder(negative)

            loss = loss_fn(anchor_encoded, positive_encoded, negative_encoded)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_triplet_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            for val_batch in val_triplet_loader:
                anchor, positive, negative = val_batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                anchor_encoded = model.encoder(anchor)
                positive_encoded = model.encoder(positive)
                negative_encoded = model.encoder(negative)

                val_loss = loss_fn(anchor_encoded, positive_encoded, negative_encoded)
                total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_triplet_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    torch.save(model, f'MEAAC.pth')