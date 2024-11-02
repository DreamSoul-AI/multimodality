import torchvision
import matplotlib.pyplot as plt
import os

root_dataset_path = 'data/mnist/'


# Downloading Mnist as image files
def load_mnist():
    if not os.path.exists(root_dataset_path):
        os.makedirs(root_dataset_path)
        train_data = torchvision.datasets.MNIST("./data", train=True, transform=torchvision.transforms.ToTensor(),
                                                download=True)
        for i, data in enumerate(train_data):
            img = data[0]
            label = data[1]
            son_path = root_dataset_path + str(label) + "/"
            if not os.path.exists(son_path):
                os.makedirs(son_path)
            plt.imsave(son_path + str(len(os.listdir(son_path))) + '.png', img.reshape(28, 28), cmap="gray")
