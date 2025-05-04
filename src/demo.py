import argparse
import logging
import numpy as np
import os
from hsi_enhancement import HSIEnhancer

def parse_arguments():
    """Parse command-line arguments for hyperparameters."""
    parser = argparse.ArgumentParser(description="HSI Resolution Enhancement Demo")
    parser.add_argument('--msi_path', type=str, default='data/Lofdal_sent.tif',
                        help='Path to MSI image file')
    parser.add_argument('--hsi_path', type=str, default='data/Lofdal_enmap.dat',
                        help='Path to HSI image file')
    parser.add_argument('--patch_size', type=int, default=12,
                        help='Size of patches for processing')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for patch processing')
    parser.add_argument('--guide_radius', type=int, default=1,
                        help='Radius for guided filter')
    parser.add_argument('--detail_weight', type=float, default=3.5,
                        help='Weight for MSI detail injection')
    parser.add_argument('--output_path', type=str, default='output/hsi_enhanced.npy',
                        help='Path to save enhanced HSI')
    return parser.parse_args()

def main():
    """Demonstrate usage of the HSIEnhancer class with configurable hyperparameters."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, filename='demo.log', filemode='w')

    # Parse command-line arguments
    args = parse_arguments()

    try:
        # Initialize and run the enhancer
        enhancer = HSIEnhancer(args.msi_path, args.hsi_path)
        hsi_enhanced = enhancer.fuse_to_enhance(
            patch_size=args.patch_size,
            stride=args.stride,
            guide_radius=args.guide_radius,
            detail_weight=args.detail_weight
        )

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        # Save the enhanced HSI
        np.save(args.output_path, hsi_enhanced)
        
        logging.info("HSI enhancement completed successfully.")
        print(f"HSI enhancement completed. Output shape: {hsi_enhanced.shape}")
        print(f"Enhanced HSI saved to: {args.output_path}")
        
        return hsi_enhanced

    except Exception as e:
        logging.error(f"Error during enhancement: {str(e)}")
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    hsi_enhanced = main()
