
training_config = {
    "TOTAL_STEPS" : 40000,
    "BATCH_SIZE"     : 8  ,
    "LEARNING_RATE"  : 1e-3,
    "NUM_EPOCHS"     : 2,
    "TRAIN_IMAGE_SHAPE" : (256, 256),
}

# style and content weights change every 10k iterations
loss_weights_config = {
    "CONTENT_WEIGHT" : 100.0,
    "STYLE_WEIGHT"  : 100,
    "TV_WEIGHT"     : 2,
}

vgg_loss_layers = {
    "CONTENT_LAYER" : 'relu4_2',
    "STYLE_LAYERS"  :('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3')
}

style_image = "style.jpg"
training_monitor_content_image = "dancing.jpg"
dataset_dir = "/kaggle/input/images256"
output_dir = "/content"
inference_content_image = "dancing.jpg"