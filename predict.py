from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# test_dir = r'.\newDataSet\test\0\00000.jpg'
test_dir = r'.\pic\10.jpg'
model = load_model('newModel.h5')
img = image.load_img(test_dir, target_size=(48, 48))
x = np.expand_dims(img, axis=0)
y = model.predict(x)

plt.figure()
x_axis = ['0 anger', '1 disgust', '2 fear', '3 happy', '4 sad', '5 surprise', '6 normal']
y_axis = np.array(y).flatten()
plt.bar(x_axis, y_axis)
plt.show()
print(np.array(y).flatten())

y = model.predict_classes(x)
print(y)
