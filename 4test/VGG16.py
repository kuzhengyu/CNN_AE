from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = VGG16(include_top=True, weights='imagenet')
# print(model.summary())

image = load_img('C:/Users/11354/Desktop/packet.jpg', target_size=(224, 224))

image_data = img_to_array(image)
image_data = np.expand_dims(image_data, axis=0)
# prepare the image data for VGG
# The only preprocessing we do is subtracting the mean RGB value, computed on the training set, from each pixel.
image_data = preprocess_input(image_data)


# using the pre-trained model to predict
prediction = model.predict(image_data)
print(prediction.shape)
# 这个就是特征

# decode the prediction results
results = decode_predictions(prediction, top=5)[0]

# print(results)

import matplotlib.pyplot as plt
#整理预测结果,value
values = []
bar_label = []
for element in results:
    values.append(element[2])
    bar_label.append(element[1])

#绘图并保存
fig=plt.figure(u"Top-5 预测结果")
ax = fig.add_subplot(111)
ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
ax.set_ylabel(u'probability')
ax.set_title(u'Top-5')
for a,b in zip(range(len(values)), values):
    ax.text(a, b+0.0005,  "%.2f"%b, ha='center', va = 'bottom', fontsize=7)

fig = plt.gcf()
plt.show()

# name=img_path[0:-4]+'_pred'
# fig.savefig(name, dpi=200)