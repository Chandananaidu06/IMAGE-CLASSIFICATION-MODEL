import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
data_dir = "clf-data"
categories = ["empty", "not_empty"]
X = []
y = []
image_paths = []

IMG_SIZE = (100, 100)  # For model input

for label, category in enumerate(categories):
    folder = os.path.join(data_dir, category)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        try:
            img = imread(img_path)
            img_resized = resize(img, IMG_SIZE)
            X.append(img_resized)
            y.append(label)
            image_paths.append(img_path)  # Save path for display
        except Exception as e:
            print(f"Error loading image {file}: {e}")

X = np.array(X)
y = np.array(y)

# Train-test split
# Train-test split
X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X, y, image_paths, test_size=0.3, random_state=42
)

# Flatten images
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# Train SVM
model = SVC()
model.fit(X_train_flat, y_train)

# Predict
y_pred = model.predict(X_test_flat)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Total samples: {len(y_test)}")
unique, counts = np.unique(y_test, return_counts=True)
print(f"Class counts: {counts}")
print(f"{acc * 100:.2f}% of samples were correctly classified.")

# Show sample prediction with original image (not resized)
idx = 0  # index of test image to show
original_img = imread(paths_test[idx])
plt.imshow(original_img)
plt.title(f"Predicted: {categories[y_pred[idx]]}")
plt.axis('off')
plt.savefig("myplot.png")  # Save image to file
plt.show()
