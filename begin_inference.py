import sys
sys.path.append('/content/real-time-neural-style-transfer')

import argparse
from config import training_config, loss_weights_config, vgg_loss_layers, dataset_dir, training_monitor_content_image, style_image, inference_content_image, output_dir
from inference import test_inference

def main():
    parser = argparse.ArgumentParser(description="train fast neural style transfer")

    parser.add_argument("--output_dir", type=str, default=output_dir, help="Directory to save results")
    parser.add_argument("--content_image", type=str, default=inference_content_image, help="content image in ineference")
    parser.add_argument("--model_path", type=str, default="model_weights", help="model weights")

    args = parser.parse_args()
    test_inference(
        args.model_path,
        args.content_image,
        args.output_dir
    )

if __name__ == "main":
    main()