import torch
import torch.nn as nn
import torch.nn.functional as F

class FaceNet(nn.Module):
    def __init__(self, embedding_size):
        super(FaceNet, self).__init__()
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, embedding_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)
        return x

def triplet_loss(anchor, positive, negative, margin=0.2):
    distance_positive = F.pairwise_distance(anchor, positive, p=2)
    distance_negative = F.pairwise_distance(anchor, negative, p=2)
    loss = torch.mean(torch.clamp(distance_positive - distance_negative + margin, min=0))
    return loss

def online_triplet_mining(model, data, labels):
    embeddings = model(data)

    anchors, positives, negatives = [], [], []
    for i in range(len(data)):
        # Select the anchor and positive samples
        anchor = data[i]
        anchor_label = labels[i]
        positive_mask = (labels == anchor_label)
        positive_samples = data[positive_mask]

        # Select the hardest negative sample
        negative_mask = (labels != anchor_label)
        negative_embeddings = embeddings[negative_mask]
        negative_distances = torch.norm(embeddings[i] - negative_embeddings, dim=1)
        negative_index = torch.argmax(negative_distances)
        negative = data[negative_mask][negative_index]

        # Add the triplet to the list
        anchors.append(anchor)
        positives.append(positive_samples[0])  # Select the first positive sample
        negatives.append(negative)

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
