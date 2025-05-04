# HSI Resolution Enhancement by MSI Fusion

This repository contains a Python library for enhancing the resolution of Hyperspectral Images (HSI) by fusing them with high-resolution Multispectral Images (MSI). The implementation uses sparse coding, wavelet transforms, and guided filtering to produce high-quality results.

## Features
- Modular code structure with separate modules for data loading, decomposition, sparse coding, patch processing, upsampling, and enhancement
- Configurable hyperparameters (patch size, stride, guide radius, detail weight) via the `fuse_to_enhance` method
- Parallel processing for efficient patch-based computations
- Advanced decomposition techniques (Wavelet, FastICA, NMF)
- Guided filtering for detail enhancement
- Demo script with command-line argument support for easy usage
- Output saving as NumPy arrays

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hsi_enhancement.git
   cd hsi_enhancement
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your MSI and HSI data files in the `data/` directory (e.g., `Lofdal_sent.tif` and `Lofdal_enmap.dat`).

2. Run the demo script with default hyperparameters:
   ```bash
   python src/demo.py
   ```

3. Customize hyperparameters via command-line arguments:
   ```bash
   python src/demo.py --msi_path data/Lofdal_sent.tif --hsi_path data/Lofdal_enmap.dat \
       --patch_size 16 --stride 2 --guide_radius 2 --detail_weight 2.5 \
       --output_path output/hsi_enhanced_custom.npy
   ```

4. Alternatively, use the library in your own scripts:
   ```python
   from hsi_enhancement import HSIEnhancer

   msi_path = 'data/Lofdal_sent.tif'
   hsi_path = 'data/Lofdal_enmap.dat'
   enhancer = HSIEnhancer(msi_path, hsi_path)
   hsi_enhanced = enhancer.fuse_to_enhance(
       patch_size=16,
       stride=2,
       guide_radius=2,
       detail_weight=2.5
   )
   print("Enhanced HSI shape:", hsi_enhanced.shape)
   ```

5. Check `demo.log` or `processing.log` for processing details. The enhanced HSI is saved to the specified output path (default: `output/hsi_enhanced.npy`).

## Project Structure
```
hsi_enhancement/
├── src/
│   ├── __init__.py         # Package initialization
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── decomposition.py    # Signal decomposition methods
│   ├── sparse_coding.py    # Sparse coding and dictionary learning
│   ├── patch_processor.py  # Patch-based processing
│   ├── upsampler.py        # HSI upsampling with MSI details
│   ├── enhancer.py         # Main HSI enhancement logic
│   ├── demo.py             # Demo script with command-line arguments
├── data/                   # Directory for input data (not included)
├── output/                 # Directory for output files (created automatically)
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── LICENSE                 # License file
└── .gitignore              # Git ignore file
```

## Requirements
See `requirements.txt` for a full list of dependencies. Key libraries include:
- NumPy
- SciPy
- Rasterio
- scikit-learn
- scikit-image
- OpenCV
- PyWavelets
- Numba
- Joblib
- TQDM

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact
For questions or support, please open an issue on GitHub.