import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle



# Data preprocessing
input_dir = "/home/user/PycharmProjects/Image-Classification/animals/animals/"
input_txt = "name of the animals.txt"
labels = []
images = []
with open(input_txt, 'r') as f:
    for value in f:
        value = value.strip('\n')
        labels.append(value)

categories = []
for index, category in enumerate(labels):
    for file in os.listdir(os.path.join(input_dir,category)):
        img_path = os.path.join(input_dir,category,file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        images.append(img.flatten())
        categories.append(index)


images = np.asarray(images)
categories = np.asarray(categories)



# Split Data
X_train, X_test, y_train, y_test = train_test_split(images, categories, test_size=0.33, shuffle=True, random_state=0)

# Model
classifier = SVC()
param_grid = [{'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}]
model = GridSearchCV(classifier, param_grid)
model.fit(X_train,y_train)


# Prediction
y_pred = model.predict(X_test)
if y_pred[0] == 0:
    print("This a cat")
else:
    print("This a dog")


# Accuracy Score
acc = accuracy_score(y_test, y_pred)
print("Accuracy Score {0}".format(acc))


# Confusion Matrix
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)