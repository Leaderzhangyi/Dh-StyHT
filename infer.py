import argparse
import torch
import torch.nn as nn
import warnings
from pathlib import Path
from model.configuration import TransModuleConfig
from model.swin import SwinTransformer
from net import HierarchiesTrans, MultiscaleVGGFusionDecoder
from utils import print_device_info, print_runtime_parameters,save_transferred_imgs,StyleTransferNet
warnings.filterwarnings('ignore')

def parse_arguments() -> argparse.Namespace:
    """
        Parse command line arguments for inference.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        default='input/dh',
        help='Input image directory containing "Content" and "Style" subdirectories'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Directory for saving output images'
    )
    parser.add_argument(
        '--checkpoint_import_path',
        type=str,
        default='pre_trained_models/weight.pkl',
        help='Path to pre-trained model weights file'
    )
    return parser.parse_args()


def get_model_config() -> TransModuleConfig:
    """
        Get the model configuration for Hierarchies Trans module.
    """
    return TransModuleConfig(
        nlayer=3,
        d_model=768,
        nhead=8,
        mlp_ratio=4,
        qkv_bias=False,
        attn_drop=0.,
        drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_first=True
    )


def build_network(device: torch.device) -> StyleTransferNet:
    """
        Build and load the Style Transfer Network.
    """
    encoder = SwinTransformer(
        img_size=224,
        patch_size=2,
        in_chans=3,
        embed_dim=192,
        depths=[2, 2, 2],
        num_heads=[3, 6, 12],
        drop_path_rate=0.1,
        patch_norm=True
    ).to(device)
    decoder = MultiscaleVGGFusionDecoder(d_model=768, seq_input=True)
    trans_module = HierarchiesTrans(get_model_config())
    return StyleTransferNet(encoder, decoder, trans_module)


def load_checkpoint(network: StyleTransferNet, checkpoint_path: str, device: torch.device) -> None:
    """
        Load pre-trained model weights from checkpoint.
    """
    print('Loading checkpoint...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.encoder.load_state_dict(checkpoint['encoder'])
    network.decoder.load_state_dict(checkpoint['decoder'])
    network.transModule.load_state_dict(checkpoint['transModule'])
    print('Checkpoint loaded successfully')
    return checkpoint.get('loss_count_interval')

def main() -> None:
    """
        Main function for executing test Dh-StyHT transfer.
    """
    args = parse_arguments()
    print_runtime_parameters(args)
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    # Set device and print info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_device_info(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build and load model
    network = build_network(device)
    load_checkpoint(network, args.checkpoint_import_path, device)
    network.to(device)
    # Execute style transfer
    save_transferred_imgs(network, args.input_dir, args.output_dir, device=device)


if __name__ == '__main__':
    main()