import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras import backend as K

#load the data, split into train and test set

(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

#reshaping the arrays

x_train  = x_train.reshape(x_train.shape[0],28,28,1)
x_test  = x_test.reshape(x_test.shape[0],28,28,1)
input_shape = (28,28,1)
num_classes = 10

#convert class vectors to binary classes

y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#feature_scaling

x_train /= 255
x_test /= 255


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#creting the CNN classifer
batch_size = 128
epochs = 10

classifier = Sequential()
#convolution-process
classifier.add(Conv2D(32,kernel_size = (3,3),activation = 'relu',input_shape = input_shape))
classifier.add(Conv2D(64,(3,3),activation = 'relu'))
#max-pooling process
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))
#flattening
classifier.add(Flatten())

#Full-connection
classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=num_classes, activation='softmax'))

#comiling the cnn
classifier.compile(optimizer = 'adadelta',loss = 'categorical_crossentropy', metrics = ['accuracy'])

#training the model

hist = classifier.fit(x_train,y_train,batch_size = batch_size,epochs = epochs,verbose = 1,validation_data = (x_test,y_test))
print('Classifer has succesfully trained!')

classifier.save('mnist.h5')
print('saving the model as mnist.h5')

score = classifier.evaluate(x_test,y_test,verbose = 0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])


















