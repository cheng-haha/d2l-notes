from d2l import torch as d2l
import torchvision

method = torchvision.transforms.RandomHorizontalFlip()
img = d2l.Image.open('cat.JPG')
method(img)
