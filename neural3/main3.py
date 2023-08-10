import numpy as np 
import matplotlib.pyplot as plt 
from tensorflow import keras
from google.colab import files
from io import BytesIO
from PIL import Image
import termios

model = keras.applications.VGG16()

uploaded = files.upload()
img = Image.open(BytesIO(uploaded[list(uploaded.keys())[0]]))
plt.imshow(img)

img = np.array(img)
x = keras.applications.vgg16.preprocess_input(img)
print(x.shape)
x = np.expand_dims(x, axis = 0)
res = model.predict(x)
print(np.argmax(res))

