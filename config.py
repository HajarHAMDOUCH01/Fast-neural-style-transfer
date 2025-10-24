
training_config = {
    "TOTAL_STEPS" : 80000,
    "BATCH_SIZE"     : 20  ,
    "LEARNING_RATE"  : 1e-3,
    "NUM_EPOCHS"     : 4,
    "TRAIN_IMAGE_SHAPE" : (256, 256),
}

# style and content weights change every 10k iterations
loss_weights_config = {
    "CONTENT_WEIGHT" : 1000.0,
    "STYLE_WEIGHT"  : 1,
    "TV_WEIGHT"     : 10,
}

vgg_loss_layers = {
    "CONTENT_LAYER" : 'relu4_2',
    "STYLE_LAYERS"  :('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3')
}

style_image = "style.jpg"
training_monitor_content_image = "dancing.jpg"
dataset_dir = "/kaggle/input/coco-2017-dataset/coco2017/train2017"
output_dir = "/content"
inference_content_image = "dancing.jpg"