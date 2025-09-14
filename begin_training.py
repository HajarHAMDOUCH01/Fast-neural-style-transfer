import sys
sys.path.append('/content/real-time-neural-style-transfer')

import argparse
from config import training_config, loss_weights_config, vgg_loss_layers, dataset_dir, training_monitor_content_image, style_image, output_dir
from train import train_style_transfer


def main():
    parser = argparse.ArgumentParser(description="train fast neural style transfer")

    parser.add_argument("--style_image", type=str, default=style_image, help="Path to style image")
    parser.add_argument("--training_monitor_content_image", type=str, default=training_monitor_content_image, help="Path to content image to monitor training")
    parser.add_argument("--dataset_dir", type=str, default=dataset_dir, help="Path to content dataset directory")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Directory to save checkpoints and final trained model and results")

    parser.add_argument("--content_weight", type=float, default=loss_weights_config['CONTENT_WEIGHT'], help="Weight for content loss")
    parser.add_argument("--style_weight", type=float, default=loss_weights_config['STYLE_WEIGHT'], help="Weight for style loss")
    parser.add_argument("--tv_weight", type=float, default=loss_weights_config['TV_WEIGHT'], help="Weight for total variation loss")

    parser.add_argument("--num_epochs", type=int, default=training_config["NUM_EPOCHS"], help="Training batch size")
    parser.add_argument("--batch_size", type=int, default=training_config["BATCH_SIZE"], help="Training batch size")
    parser.add_argument("--total_steps", type=int, default=training_config['TOTAL_STEPS'], help="Total training steps")
    parser.add_argument("--lr", type=float, default=training_config['LEARNING_RATE'], help="Learning rate")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="start training from a checkpoint")


    args = parser.parse_args()

    # if training from the start : 
    train_style_transfer(
        args.style_image,
        args.training_monitor_content_image,
        args.dataset_dir,
        args.output_dir,
        args.content_weight,
        args.style_weight,
        args.tv_weight,
        args.num_epochs,
        args.batch_size,
        args.total_steps,
        args.lr,
        args.checkpoint_path
    )

if __name__ == '__main__':
    main()