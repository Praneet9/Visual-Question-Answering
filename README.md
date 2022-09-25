# Visual-Question-Answering

## Setting up for Inference

### Docker
```bash
$ git clone https://github.com/Praneet9/Visual-Question-Answering.git
$ cd Visual-Question-Answering

# Build the docker
$ docker build -t vqa .

# Run the docker with port 7860 to access the app
$ docker run --gpus all -it -p 7860:7860 vqa
```

### Native
```bash
$ sudo apt install git wget unzip -y
$ git clone https://github.com/Praneet9/Visual-Question-Answering.git
$ cd Visual-Question-Answering

# Installing the required python packages
$ pip3 install torch timm pyyaml gradio tqdm

# Downloading model weights and vocab+embeddings
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget \
    --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
    'https://docs.google.com/uc?export=download&id=1bJQQ3pMm58xRDnBHapQ39r3Z_nloVHEH' -O- | \
    sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bJQQ3pMm58xRDnBHapQ39r3Z_nloVHEH" \
    -O vqa_inference_data.zip && rm -rf /tmp/cookies.txt

# Unzipping trained model and embeddings
$ unzip -q vqa_inference_data.zip

# Downloading pretrained feature extractors
$ python3 -c "import timm; timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0); \
    timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, global_pool='');"

# Run the app
$ python3 app.py
```

**Note: You can also download the trained model and vocab+embeddings from [here](https://drive.google.com/file/d/1bJQQ3pMm58xRDnBHapQ39r3Z_nloVHEH/view?usp=sharing)**

Open `http://localhost:7860` in your browser to access the app for inference.

---

## Setting up for Training

```bash
$ sudo apt install git wget unzip -y
$ git clone https://github.com/Praneet9/Visual-Question-Answering.git
$ cd Visual-Question-Answering

# Installing all the dependencies
$ pip3 install -r requirements.txt

# Downloading VQA v2 dataset and glove embeddings
$ ./download_files.sh

# Converting glove embeddings to readable numpy arrays
$ python3 convert_emb_to_np.py

# Save ViT features of images for training and validation
$ python3 save_vit_features.py --data_type train_data
$ python3 save_vit_features.py --data_type validation_data

# Start model training
$ python3 trainer.py
```

**Note: Downloading dataset, Saving ViT features and model training is very time intensive.**

Happy Training! :)

---

## References

- [VisualQA v2 dataset](https://visualqa.org/)
- [Glove pretrained vectors](https://nlp.stanford.edu/projects/glove/)
- [Pretrained ViT model](https://github.com/rwightman/pytorch-image-models)
- [VQA Paper](https://arxiv.org/pdf/1612.00837.pdf)


## Citation

```
@InProceedings{balanced_vqa_v2,
author = {Yash Goyal and Tejas Khot and Douglas Summers{-}Stay and Dhruv Batra and Devi Parikh},
title = {Making the {V} in {VQA} Matter: Elevating the Role of Image Understanding in {V}isual {Q}uestion {A}nswering},
booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2017},
}
```