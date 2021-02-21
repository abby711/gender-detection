from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D,Activation,Flatten,Dropout,Dense
from tensorflow.keras import backend
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random,cv2,os,glob

e=5
lr=1e-3
batchsize=64
dims=(96,96,3)

data=[]
labels=[]

imagefiles=[f for f in glob.glob(r'dataset'+ "/**/*",recursive=True) if not os.path.isdir(f)]
random.shuffle(imagefiles)

for img in imagefiles:
    image=cv2.imread(img)
    image=cv2.resize(image,(dims[0],dims[1]))
    image=img_to_array(image)
    data.append(image)
    label=img.split(os.path.sep)[-2] #either from woman or men directory
    if label== "woman":
        label=1
    else:
        label=0
    labels.append([label])

data=np.array(data,dtype="float")/255.0
labels=np.array(labels)

(trainx,testx,trainy,testy)=train_test_split(data,labels,test_size=0.2,random_state=42) #205 data for test 80% data for train

trainy=to_categorical(trainy,num_classes=2) #if man then [1,0] else if womn [0,1]
testy=to_categorical(testy,num_classes=2)

aug=ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")   # to geneate various image forms like rotating mirroe imaging etc

def build(w,h,d,classes):
    model=Sequential()
    inputshape=(h,w,d)
    chandim=-1

    if backend.image_data_format() == "channels_first": #Returns a string, either 'channels_first' or 'channels_last'
        inputshape = (d, h, w)
        chandim = 1

    # The axis that should be normalized, after a Conv2D layer with data_format="channels_first",
    # set axis=1 in BatchNormalization.

    model.add(Conv2D(32, (3,3), padding="same", input_shape=inputshape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chandim))
    model.add(MaxPooling2D(pool_size=(3,3))) #to avoid unwanted noice
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chandim))

    model.add(Conv2D(64, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chandim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chandim))

    model.add(Conv2D(128, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chandim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model


model = build(w=dims[0], h=dims[1], d=dims[2],
                            classes=2)


opt = Adam(lr=lr, decay=lr/e)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model.fit_generator(aug.flow(trainx, trainy, batch_size=batchsize),
                        validation_data=(testx,testy),
                        steps_per_epoch=len(trainx) // batchsize,
                        epochs=e, verbose=1)
print("********8saving model to diskk**************")
model.save('genderdetection.model')

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = e
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig('plot.png')
