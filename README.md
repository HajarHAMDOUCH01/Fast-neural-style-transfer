# Fast Neural Style Transfer

[Try model (Streamlit inference)](https://real-time-nst-app.streamlit.app/)

A PyTorch implementation of the fast neural style transfer method from the paper ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/abs/1603.08155) by Johnson et al. (2016), using Instance Normalization instead of Batch Normalization.

This implementation trains a feed-forward convolutional neural network to transform images in the style of a given artwork, achieving real-time style transfer (~3 seconds on a T4 GPU in Google Colab).

## Overview

**Original Image**

[![image alt](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/raw/2d84ee10d6b86239ce1e406fb1c43c425cb75e1a/dancing.jpg)](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/blob/2d84ee10d6b86239ce1e406fb1c43c425cb75e1a/dancing.jpg)

**First style: Picasso Art**

[![image alt](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/raw/084646b569054c614720d8533f28a1e73ac1c2e9/picasso.jpg)](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/blob/084646b569054c614720d8533f28a1e73ac1c2e9/picasso.jpg)

**Stylized Image**

[![image alt](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/raw/c75f0f596035fb28c302b4152a537e6973f6a81f/sample_image_picasso.jpg)](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/blob/c75f0f596035fb28c302b4152a537e6973f6a81f/sample_image_picasso.jpg)

The method combines two key innovations:
- **Feed-forward transformation network**: A deep CNN that learns to transform images in a single forward pass
- **Perceptual loss functions**: Loss functions based on high-level features from a pretrained VGG-19 network rather than pixel-wise differences

## Architecture

### Transformation Network
- **Input**: RGB images of shape 3×256×256
- **Output**: Stylized RGB images of same dimensions
- **Architecture**:
  - 2 strided convolutions for downsampling (stride=2)
  - 5 residual blocks for feature transformation
  - 2 fractionally-strided convolutions for upsampling (stride=1/2)
  - Instance Normalization and ReLU activations throughout
  - Final scaled tanh to ensure output pixels in range [0,255]

### Loss Network
- **Pretrained VGG-19** (frozen during training)
- Used to extract features for perceptual loss computation
- No updates to VGG-19 weights during style transfer training

## Loss Functions

The training objective combines three loss terms:

### 1. Feature Reconstruction Loss
Encourages content preservation by matching high-level features:

[![image alt](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/raw/732c3d0faf1f46c5b2630760d1027df3f0d243e1/feature_reconstruction_loss.png)](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/blob/732c3d0faf1f46c5b2630760d1027df3f0d243e1/feature_reconstruction_loss.png)

- Computed at VGG19 layer `relu4_2`
- φ_j(x) represents activations at layer j

### 2. Style Reconstruction Loss
Captures style characteristics using Gram matrix correlations:

[![image alt](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/raw/073233a7b891a6d187c13c128c789560f948dc8e/style_loss.png)](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/blob/073233a7b891a6d187c13c128c789560f948dc8e/style_loss.png)

- Computed at VGG layers: `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, `relu5_1`
- You can choose higher layers to capture a different representation of style, or weight the style layers individually.

### 3. Total Variation Regularization
Promotes spatial smoothness in output:

[![image alt](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/raw/8a4f6522246861b5f622497e1b8880542b0fc5d2/tv_loss.png)](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/blob/8a4f6522246861b5f622497e1b8880542b0fc5d2/tv_loss.png)

### Combined Loss

[![image alt](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/raw/940f0c59c4473ad65008bb25af1adb7399918682/total_loss.png)](https://github.com/HajarHAMDOUCH01/real-time-neural-style-transfer/blob/940f0c59c4473ad65008bb25af1adb7399918682/total_loss.png)

## Dataset and Training

### Training Data
- **COCO 2017 Dataset**: ~118k training images
- Images resized to 256×256 pixels
- Training for ~2 epochs (sufficient to avoid overfitting), 40k steps total

### Training Parameters
- **Batch size**: 4
- **Optimizer**: Adam with learning rate 1×10⁻³
- **Iterations**: 40,000 (approximately 2 epochs)
- **Loss weights**:
  - λ_c = 1.0 (content weight)
  - λ_s = 100.0 (style weight for starry night)
  - λ_TV = 1×10⁻¹ to 1×10⁻² (chosen via cross-validation per style)

### Training Process
1. Load pretrained VGG-19 and freeze weights
2. Initialize transformation network randomly
3. For each batch:
   - Forward pass through transformation network
   - Compute perceptual losses using VGG-19 features (`relu4_2` as content layer; `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, `relu5_1` as style layers, equally weighted)
   - Backpropagate and update transformation network only
4. Save model after convergence

## Key Features

### Real-time Performance
- **Speed**: Processes 256×256 images in ~3 seconds on a Google Colab T4 GPU
- **Efficiency**: ~1000× faster than optimization-based methods
- **Quality**: Comparable results to iterative optimization

### Generalization
- Networks trained on 256×256 images work on arbitrary resolutions
- Fully convolutional architecture enables variable input sizes
- Single forward pass regardless of image size

## Usage

**Install requirements**
```
pip install -r requirements.txt
```

**Training**

Default parameters are in `config.py`
```
python begin_training.py \
    --style_image "styles/starry_night.jpg" \
    --training_monitor_content_image "content/mountains.jpg" \
    --dataset_dir "data/train_images" \
    --output_dir "results/checkpoints_and_images_and_final_weights" \
    --content_weight 1e5 \
    --style_weight 1e10 \
    --tv_weight 1e-6 \
    --num_epochs 2 \
    --batch_size 4 \
    --total_steps 40000 \
    --lr 1e-3
```

**Inference**
```
python begin_inference.py \
    --model_path "checkpoints/style_transfer_model.pth" \
    --content_image "content/mountains.jpg" \
    --output_dir "results/stylized_image"
```

## Implementation Notes

### Architectural Choices
- **Residual connections**: Help preserve image structure during transformation
- **Downsampling strategy**: Reduces computational cost and increases receptive field
- **Instance normalization**: Gives better results than batch normalization for this task

## Requirements

- PyTorch ≥ 1.7
- torchvision
- PIL/Pillow
- NumPy
- CUDA-capable GPU (recommended)

## Contributing

We welcome contributions to this implementation! Here's how you can help:

**Ways to Contribute**
- *Bug fixes*: Report and fix issues in the training or inference pipeline
- *Performance improvements*: Optimize code for better speed or memory usage
- *Documentation*: Improve README, add code comments, or create tutorials
- *Testing*: Add unit tests, integration tests, or performance benchmarks
- *Features*: Implement additional loss functions or network architectures

**Contribution Guidelines**
1. Fork the repository and create a feature branch
2. Follow coding standards: use consistent formatting and meaningful variable names
3. Test your changes: ensure your code works with the existing pipeline
4. Document your work: add docstrings and update the README if needed
5. Submit a pull request with a clear description of changes

**Development Setup**
```
git clone https://github.com/HajarHAMDOUCH01/Fast-neural-style-transfer
cd Fast-neural-style-transfer
pip install -r requirements.txt
```

Please ensure all contributions maintain compatibility with the original paper's methodology and produce comparable results.

## References
```
@article{johnson2016perceptual,
  title={Perceptual losses for real-time style transfer and super-resolution},
  author={Johnson, Justin and Alahi, Alexandre and Fei-Fei, Li},
  journal={arXiv preprint arXiv:1603.08155},
  year={2016}
}
```

## License
This implementation is for research and educational purposes. Please refer to the original paper for detailed methodology and theoretical background.
