
training_config = {
    "TOTAL_STEPS" : 40000,
    "BATCH_SIZE "     : 4  ,
    "LEARNING_RATE "  : 1e-2,
    "NUM_EPOCHS "     : 2,
    "TRAIN_IMAGE_SHAPE" : (256, 256),
}

# for 'scream' style , every style has different config for weights - to be done later 
loss_weights_config = {
    "CONTENT_WEIGHT" : 1.0,
    "STYLE_WEIGHT "  : 60.0,
    "TV_WEIGHT "     : 1e-2,
}

vgg_loss_layers = {
    "CONTENT_LAYER" : 'relu4_2',
    "STYLE_LAYERS"  :('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
}
