"""
Utility functions for image processing and model operations.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import zipfile
import os

@dataclass
class ColorCodes:
    """ANSI color codes for terminal output"""
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    END = '\033[0m'

def print_device_info(device: torch.device) -> None:
    """Print formatted device information with color and styling."""
    c = ColorCodes()
    device_info = f"{c.BOLD}{c.GREEN}Using device: {device}{c.END}"
    
    if torch.cuda.is_available():
        gpu_info = f"{c.BOLD}{c.BLUE}GPU: {torch.cuda.get_device_name(0)}{c.END}"
        print(f"{device_info}, {gpu_info}")
    else:
        print(device_info)

def print_runtime_parameters(args: Dict) -> None:
    """Print formatted runtime parameters with color and styling."""
    c = ColorCodes()
    print(f"\n{c.BOLD}{c.CYAN}Runtime Configuration:{c.END}")
    print(f"{c.YELLOW}{'=' * 50}{c.END}")
    
    for key, value in sorted(vars(args).items()):
        param_name = f"{c.BOLD}{key}{c.END}"
        param_value = f"{c.CYAN}{value}{c.END}"
        print(f"{param_name:.<50} {param_value}")
    
    print(f"{c.YELLOW}{'=' * 50}{c.END}\n")

def show_torch_image(image: torch.Tensor) -> None:
    """Display a PyTorch tensor as an image."""
    if len(image.shape) == 4:
        image = image.squeeze(0)
    
    plt.imshow(T.ToPILImage()(image))
    plt.axis('off')
    plt.show()
    plt.close()

def zip_directory(zip_path: Path, dir_path: Path) -> None:
    """Compress a directory into a ZIP file."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(dir_path))

def load_image_to_tensor(
    image_path: Path,
    transform: Optional[T.Compose] = None
) -> torch.Tensor:
    """Load an image and convert it to a PyTorch tensor."""
    transform = transform or T.ToTensor()
    image = Image.open(image_path)
    return transform(image).unsqueeze(0)

def prepare_content_style_images(
    content_path: Path,
    style_path: Path,
    img_size: int = 224
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare content and style images for processing."""
    content_transform = T.Compose([
        T.Resize((img_size, img_size), Image.BICUBIC),
        T.RandomCrop(img_size),
        T.ToTensor()
    ])
    
    style_transform = T.Compose([
        T.Resize((512, 512), Image.BICUBIC),
        T.RandomCrop(img_size),
        T.ToTensor()
    ])
    
    content_img = load_image_to_tensor(content_path, content_transform)
    style_img = load_image_to_tensor(style_path, style_transform)
    
    return content_img, style_img

# @torch.no_grad()
# def save_arbitrary_size_samples(
#     network: nn.Module,
#     samples_dir: Path,
#     output_dir: Path,
#     device: torch.device = torch.device('cpu')
# ) -> None:
#     """Test and save sample images with arbitrary size."""
#     sample_pairs = {
#         '1': range(1, 12),
#         '2': range(1, 12),
#     }
    
#     print('Starting image generation:')
#     for content_idx, style_indices in tqdm(sample_pairs.items()):
#         output_images = []
#         content_path = samples_dir / f'Content/{content_idx}.jpg'
        
#         for style_idx in style_indices:
#             style_path = samples_dir / f'Style/{style_idx}.jpg'
#             content, style = prepare_content_style_images(content_path, style_path)
            
#             result = network(content.to(device), style.to(device), arbitrary_input=True)
#             style = TF.center_crop(style, content.shape[2:])
#             result = TF.center_crop(result, content.shape[2:])
            
#             output_images.extend([content.cpu(), style.cpu(), result.cpu()])
        
#         output_tensor = torch.cat(output_images, dim=0)
#         save_image(output_tensor, output_dir / f'test_{content_idx}.jpg', nrow=3)

def get_train_transform() -> T.Compose:
    """Get transform pipeline for training images."""
    return T.Compose([
        T.Resize((224, 224), Image.BICUBIC),
        T.RandomCrop(224),
        T.ToTensor()
    ])

@torch.no_grad()
def calculate_average_generation_time(
    network: nn.Module,
    content_path: Path,
    style_path: Path,
    rounds: int = 1,
    device: torch.device = torch.device('cpu')
) -> float:
    """Calculate average time for image generation."""
    content = load_image_to_tensor(content_path).to(device)
    style = load_image_to_tensor(style_path).to(device)
    start_time = datetime.now()
    for _ in range(rounds):
        network(content, style, arbitrary_input=True)
    end_time = datetime.now()
    time_diff = (end_time - start_time).total_seconds()
    return time_diff / rounds

@torch.no_grad()
def calculate_batch_generation_time(
    network: nn.Module,
    samples_dir: Path,
    device: torch.device = torch.device('cpu')
) -> float:
    """Calculate average generation time for multiple images."""
    content_files = list((samples_dir / 'Content').glob('*'))
    style_files = list((samples_dir / 'Style').glob('*'))
    total_pairs = len(content_files) * len(style_files)
    start_time = datetime.now()
    for content_path in content_files:
        for style_path in style_files:
            content, style = prepare_content_style_images(content_path, style_path)
            network(content.to(device), style.to(device), arbitrary_input=True)
    end_time = datetime.now()
    time_diff = (end_time - start_time).total_seconds()
    return time_diff / total_pairs

@torch.no_grad()
def save_transferred_imgs(
    network: nn.Module,
    samples_dir: Path,
    output_dir: Path,
    device: torch.device = torch.device('cpu')
) -> None:
    """Transfer style between content and style images and save results.
    
    Args:
        network: Neural network model for style transfer
        samples_dir: Directory containing 'content' and 'style' subdirectories
        output_dir: Directory to save generated images
        device: Device to run the model on
    """
    print('Starting image generation:')
    samples_dir = Path(samples_dir)
    output_dir = Path(output_dir)
    # Get all content and style image paths
    content_dir = samples_dir / 'content'
    style_dir = samples_dir / 'style'
    content_paths = list(content_dir.glob('*'))
    style_paths = list(style_dir.glob('*'))
    # Process each content-style pair
    for content_path in tqdm(content_paths, desc='Processing content images'):
        for style_path in tqdm(style_paths, desc='Applying styles', leave=False):
            # Prepare input images
            content_img, style_img = prepare_content_style_images(content_path, style_path)
            # Generate styled image
            styled_img = network(
                content_img.to(device),
                style_img.to(device),
                arbitrary_input=True
            )
            # Create output filename
            content_stem = content_path.stem
            style_stem = style_path.stem
            output_suffix = content_path.suffix
            output_name = f'{content_stem} + {style_stem}{output_suffix}'
            output_path = output_dir / output_name
            # Save generated image
            save_image(styled_img, output_path)

class StyleTransferNet(nn.Module):
    """Neural network for style transfer.
    
    Attributes:
        encoder: Encoder network for feature extraction
        decoder: Decoder network for image reconstruction
        transModule: Transformer module for style transfer
        patch_size: Size of image patches for processing
    """
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        transModule: nn.Module,
        patch_size: int = 8
    ) -> None:
        """Initialize StyleTransferNet.
        
        Args:
            encoder: Encoder network
            decoder: Decoder network
            transModule: Transformer module
            patch_size: Size of image patches (default: 8)
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transModule = transModule
        self.patch_size = patch_size

    def forward(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        arbitrary_input: bool = False
    ) -> torch.Tensor:
        """Forward pass for style transfer.
        Args:
            content_img: Content image tensor
            style_img: Style image tensor
            arbitrary_input: Flag for arbitrary size input processing
            
        Returns:
            Styled image tensor
        """
        # Get input dimensions
        _, _, height, width = content_img.size()
        # Set decoder dimensions
        self.decoder.img_H = height
        self.decoder.img_W = width
        # Extract features
        content_features = self.encoder(content_img)
        style_features = self.encoder(style_img)
        # Calculate feature resolution
        num_patches = content_features.size(1)
        feature_resolution = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        # Apply style transfer
        transferred_features = self.transModule(content_features, style_features)
        # Generate output image
        output_img = self.decoder(transferred_features, feature_resolution)
        return output_img