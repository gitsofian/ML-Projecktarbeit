
from cifar_loader import CIFAR10
import matplotlib.pyplot as plt
import numpy as np

# Lade CIFAR10 Datensatz
cifar10 = CIFAR10('cifar10')

# Lade Trainings und Testdaten, sowie die Labels der Klassen
images_train, labels_train = cifar10.train_data, cifar10.train_labels
images_test, labels_test = cifar10.test_data, cifar10.test_labels
label_names = cifar10.label_names

# Plotte Beispiele um den Datensatz zu visualisieren
fig, axes = plt.subplots(1, 10, sharey=True)
for i, ax in enumerate(axes):
    subset = np.asarray(labels_train == i).nonzero()[0]
    idx = np.random.choice(subset)
    image = images_train[idx].reshape((3, 32, 32)).transpose((1,2,0))
    label = labels_train[idx]
    
    ax.imshow(image)
    ax.set_title(label_names[i])
    ax.axis("off")
plt.show()

# --- Projektarbeit ---