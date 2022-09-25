FROM pytorch/pytorch:latest

RUN apt update -y && apt install git wget unzip -y
RUN pip3 install --upgrade pip && pip3 install timm pyyaml gradio tqdm

# Downloading model weights and vocab+embeddings
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget \
    --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
    'https://docs.google.com/uc?export=download&id=1bJQQ3pMm58xRDnBHapQ39r3Z_nloVHEH' -O- | \
    sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bJQQ3pMm58xRDnBHapQ39r3Z_nloVHEH" \
    -O vqa_inference_data.zip && rm -rf /tmp/cookies.txt

# Downloading pretrained feature extractors
RUN python3 -c "import timm; timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0); \
                timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, global_pool='');"

RUN mkdir /Visual-Question-Answering && mv vqa_inference_data.zip /Visual-Question-Answering/
COPY . /Visual-Question-Answering/
WORKDIR /Visual-Question-Answering

# Unzipping trained model and embeddings
RUN unzip -q vqa_inference_data.zip

CMD ["python3", "app.py"]