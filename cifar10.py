from cifar_loader import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


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
    image = images_train[idx].reshape((3, 32, 32)).transpose((1, 2, 0))
    label = labels_train[idx]

    ax.imshow(image)
    ax.set_title(label_names[i])
    ax.axis("off")

# --- Projektarbeit ---
# K-Neigh-board Klasifisierung Methode

# X = images_train
# y = labels_train

# Aufgabe a) --> TRAINING
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(images_train, labels_train)

# Aufgabe b) --> PREDICTION
pred_test = clf.predict(images_test)

# Aufgabe c) -->
print(f"score : {clf.score(images_test, labels_test)}")
conf = confusion_matrix(labels_test, pred_test)


if conf is not None:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(conf, cmap="YlGn")
    # ax.axis("off")
    ax.set_xlabel("prediction")
    ax.set_ylabel("ground truth")
    ax.set_xticks(label_names)
    ax.set_yticks(label_names)

    for i in range(10):
        for j in range(10):
            ax.text(j, i, "%d" % conf[i, j], color="black",
                    horizontalalignment='center', verticalalignment='center')


plt.show()
