from keras.models import load_model
import numpy as np
from keras.preprocessing import image

model = load_model('DogCat.h5')


test_img = image.load_img('single_prediction/dog.jpg',target_size=(64,64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img,axis = 0 )

result = model.predict(test_img)

if result[0][0] ==1:
    prediction = 'Dog'
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)
