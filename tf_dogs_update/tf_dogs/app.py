import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np




# 1. 加载模型
model = load_model(r'D:\dog breeds\model\resnet_model.h5')


# 2. 加载图像进行预处理
img_path = r'D:\dog breeds\test\image.jpg'  # 替换为你要用于推断的图像路径


# 3. 进行推断
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # 归一化像素值

predictions = model.predict(img_array)[0]  # 假设 batch size 为 1

# 获取最有可能的前 k 个品种
top_k = 5
top_indices = np.argsort(predictions)[-top_k:][::-1]

# 输出各个品种的标签和概率
for i in top_indices:
    label = class_labels[i]  # 替换为你的类别标签列表
    probability = predictions[i]
    print(f"{label}: {probability:.2%}")