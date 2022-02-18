
from load_input_data import *
from build_model import *
from tensorflow.keras.losses import sparse_categorical_crossentropy

# Initials
no_classes = 131
input_shape = (100, 100, 3)
model_no = 1000

# Model configuration
batch_size = 400
loss_function = sparse_categorical_crossentropy
no_epochs = 60
# FILES = "../"

train_data, validation_data = load_input_data(batch_size=batch_size, image_size=input_shape[:-1])
tf.keras.backend.clear_session()

for i in range(model_no):
    randInt = []
    model, randInt = define_model(randInt=randInt,
                no_classes = no_classes, 
                input_shape=input_shape)
    history = model.fit(
        train_data, 
        epochs = no_epochs,
        steps_per_epoch = train_data.samples // batch_size,
        validation_data = validation_data, 
        validation_steps = validation_data.samples // batch_size,
    )
    tempStr = ""
    for j in randInt:
        tempStr += str(j) + "-"
    modelName = "000st-"+str(i)+"s-"+str(history.history['accuracy'][-1])+"-l-"+tempStr
    model.save("Models/"+modelName)    
