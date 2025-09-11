# Fast Neural Style Transfer

A PyTorch implementation of the fast neural style transfer method from the paper ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution"](https://arxiv.org/abs/1603.08155) by Johnson et al. (2016).

This implementation trains a feed-forward convolutional neural network to transform images in the style of a given artwork, achieving real-time style transfer that is three orders of magnitude faster than optimization-based methods.

## Overview

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
  - Batch normalization and ReLU activations throughout
  - Final scaled tanh to ensure output pixels in range [0,255]

### Loss Network
- **Pretrained VGG-19** (frozen during training)
- Used to extract features for perceptual loss computation
- No updates to VGG-19 weights during style transfer training

## Loss Functions

The training objective combines three loss terms:

### 1. Feature Reconstruction Loss
Encourages content preservation by matching high-level features:
```
ℓ_feat^φ,j(ŷ,y) = 1/(C_j H_j W_j) ||φ_j(ŷ) - φ_j(y)||²₂
```
- Computed at VGG19 layer `relu4_2`
- φ_j(x) represents activations at layer j

### 2. Style Reconstruction Loss
Captures style characteristics using Gram matrix correlations:
```
ℓ_style^φ,j(ŷ,y) = ||G^φ_j(ŷ) - G^φ_j(y)||²_F
```
Where Gram matrix: `G^φ_j(x)_c,c' = 1/(C_j H_j W_j) Σ_h,w φ_j(x)_h,w,c φ_j(x)_h,w,c'`
- Computed at VGG layers: `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, `relu5_1`

### 3. Total Variation Regularization
Promotes spatial smoothness in output:
```
ℓ_TV(ŷ) = Σ_i,j [(ŷ_i,j+1 - ŷ_i,j)² + (ŷ_i+1,j - ŷ_i,j)²]
```

### Combined Loss
```
L = λ_c ℓ_feat + λ_s Σ_j ℓ_style^j + λ_TV ℓ_TV
```

## Dataset and Training

### Training Data
- **COCO 2017 Dataset**: ~118k training images
- Images resized to 256×256 pixels
- Training for ~2 epochs (sufficient to avoid overfitting) in 40k steps

### Training Parameters
- **Batch size**: 4
- **Optimizer**: Adam with learning rate 1×10⁻³
- **Iterations**: 40,000 (approximately 2 epochs)
- **Loss weights**:
  - λ_c = 1.0 (content weight)
  - λ_s = 60.0 (style weight for starry night)
  - λ_TV = 1×10⁻1 to 1×10⁻2 (chosen via cross-validation per style)

### Training Process
1. Load pretrained VGG-19 and freeze weights
2. Initialize transformation network randomly
3. For each batch:
   - Forward pass through transformation network
   - Compute perceptual losses using VGG-19 features (from relu4_2 as content layer and relu1_1, relu2__1, relu3_1, relu4_1, relu5_1 as style layers with the same weight per layer)
   - Backpropagate and update transformation network only
4. Save model after convergence

## Key Features

### Real-time Performance
- **Speed**: Processes 256×256 images at 4 seconds on google colab T4 GPU
- **Efficiency**: 1000× faster than optimization-based methods
- **Quality**: Comparable results to iterative optimization

### Generalization
- Networks trained on 256×256 images work on arbitrary resolutions
- Fully convolutional architecture enables variable input sizes
- Single forward pass regardless of image size

## Usage

**Instal requirements**
```python
pip install -r requirements.txt
```

**Training** 
Default parameters are in config.py
```python
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
```python
python begin_inference.py \
    --model_path "checkpoints/style_transfer_model.pth" \
    --content_image "content/mountains.jpg" \
    --output_dir "results/stylized_image"
```

## Implementation Notes

### Architectural Choices
- **Residual connections**: Help preserve image structure during transformation
- **Downsampling strategy**: Reduces computational cost and increases receptive field
- **Batch normalization**: Stabilizes training and improves convergence
- **Fractional stride convolutions**: Learnable upsampling vs fixed interpolation

### Training Stability
- **Batch normalization**: Essential for stable training
- **Adam optimizer**: Better convergence than SGD for this architecture
- **Learning rate**: 1×10⁻³ works well across different styles
- **weight decay**: weight decay with CosineAnnealingLR

## Requirements

- PyTorch ≥ 1.7
- torchvision
- PIL/Pillow
- NumPy
- CUDA-capable GPU (recommended)

## Coming Soon
**Stay tuned for:**

- Inference implementation using Transformers in Hugging Face
- Comprehensive performance tests and benchmarks
- Quantitative results on standard test datasets
- Speed comparisons across different hardware configurations

## Contributing
We welcome contributions to this implementation! Here's how you can help:
Ways to Contribute

*Bug fixes*: Report and fix issues in the training or inference pipeline
*Performance improvements*: Optimize code for better speed or memory usage
*Documentation*: Improve README, add code comments, or create tutorials
*Testing*: Add unit tests, integration tests, or performance benchmarks
*Features*: Implement additional loss functions or network architectures

**Contribution Guidelines**

Fork the repository and create a feature branch
Follow coding standards: Use consistent formatting and meaningful variable names
Test your changes: Ensure your code works with the existing pipeline
Document your work: Add docstrings and update README if needed
Submit a pull request with clear description of changes

Development Setup
```python
bashgit clone https://github.com/HajarHAMDOUCH01/fast-neural-style-transfer
cd fast-neural-style-transfer
pip install -r requirements.txt
```

Please ensure all contributions maintain compatibility with the original paper's methodology and produce comparable results.

## References

bibtex@article{johnson2016perceptual,
  title={Perceptual losses for real-time style transfer and super-resolution},
  author={Johnson, Justin and Alahi, Alexandre and Fei-Fei, Li},
  journal={arXiv preprint arXiv:1603.08155},
  year={2016}
}

## License
This implementation is for research and educational purposes. Please refer to the original paper for detailed methodology and theoretical background.
