<center><h1>DH-StyHT: Dunhuang Art Style Transfer via Hierarchical Vision Transformer and Color Consistency Constraints</h1></center>


# Introduction
This is the official implementation of the paper "DH-StyHT: Dunhuang Art Style Transfer via Hierarchical Visual Transformers and Color Consistency Constraints". This framework aims to meet the needs of promoting Dunhuang mural art in a modern context. This repository provides a pre-trained model for you to generate your own images based on content images and style images. In addition, you can download the publicly available dataset and weights from Google Drive.

<div align="center">
  <img src="imgs/differentStyle.png" height="auto">
</div>

We agree that there are certain commonalities between artistic styles in terms of content and color expression, but we also note that some specific artistic styles (such as Dunhuang murals) have significant differences in fine-grained features, especially in terms of color balance and texture expression. It is very meaningful to explore a suitable Dunhuang style art migration framework.

# Style Transfer Examples
<div align="center">
  <img src="imgs/compare.jpg" height="auto">
  <p>Stylized results.</p>
</div>

<div align="center">
  <img src="imgs/example.jpg" width="50%" height="auto">
  <p>Different images and different effects.</p>
</div>


# TODO

- [X] Release the inference code.
- [ ] Release training data.


## Requirements

We recommend using [Conda](https://www.anaconda.com/about-us) to manage dependencies. Run these commands from the repo root and create a new environment with all dependencies installed.

```bash 
conda env create -f environment.yml -n DhStyHT   # Create environment
conda activate DhStyHT                           # Activate environment
```

Some of the data has been previously made available as open source, please see the [links](https://drive.google.com/file/d/1zqFX_gg6Pp4kf4PrmKB7NIojQDSxS3xr/view) for specific information.


## Inference

1) Download the pretrained [model](https://drive.google.com/file/d/1GDJPWTapKQlRwfcwzdxFqeyWVMY7xhiv/view?usp=sharing) from  and put it under `./pre_trained_models/`.
2) Prepare content and style images and place them in the `dh/content` and `dh/style` folders respectively.
```bash
└── dh
    ├── content
    │   ├── xx.png
    │   └── xx.png
    └── style
        ├── xx.png
        └── xx.png
```
3) Run the commands in terminal. Then you can view the generated images in the result folder.
```python3
python3 infer.py --input_dir dh --output_dir ./results --checkpoint_import_path pre_trained_models/weight.pkl
```

# Citation

