from torchvision import transforms
import timm

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.82755924, 0.73666559, 0.8148398), (0.20211266, 0.23757748, 0.18255166)),
])

