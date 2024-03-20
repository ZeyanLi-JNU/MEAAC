import random
from torchvision.transforms import functional as F
from sklearn.decomposition import PCA
import numpy as np
from PIL import Image

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, height, width = image.shape

            image = image.flip(-1)

            if "boxes" in target:
                bbox = target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox

            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)

        return image, target



class PCATransform:
    def __init__(self, n_components, copy=True, whiten=False):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.pca = PCA(n_components=n_components, copy=copy, whiten=whiten)

    def __call__(self, img, target=None):
        img_array = np.array(img)
        original_shape = img_array.shape
        img_array = img_array / 255.0
        img_array = img_array.reshape(-1, img_array.shape[-1])
        img_array = img_array - np.mean(img_array, axis=0)
        img_array = self.pca.fit_transform(img_array)
        img_array = img_array.reshape(original_shape[0], original_shape[1], self.n_components)
        img = Image.fromarray((img_array * 255).astype('uint8'))
        img = F.to_tensor(img)
        return img, target