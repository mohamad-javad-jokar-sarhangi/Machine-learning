from keras.utils import plot_model
from keras.models import load_model
# مدل ذخیره شده را بخوان
model = load_model(r'prerequisite\Run on CPU GPU\19 tir GPU Level 3\02 Model CNN 1 with GPU.h5')

# یک عکس از معماری مدل بساز
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
