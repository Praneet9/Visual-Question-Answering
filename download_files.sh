#! /usr/bin/env bash

echo "********************************** Downloading Embeddings... **********************************"
wget -P embeddings/ http://nlp.stanford.edu/data/wordvecs/glove.6B.zip

echo "********************************** Unzipping Embeddings... **********************************"
unzip -q embeddings/glove.6B.zip -d embeddings/

echo "********************************** Downloading Annotations... **********************************"
wget -P dataset/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget -P dataset/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

echo "********************************** Unzipping Annotations... **********************************"
unzip -q dataset/v2_Annotations_Train_mscoco.zip -d dataset/
unzip -q dataset/v2_Annotations_Val_mscoco.zip -d dataset/

echo "********************************** Downloading Questions... **********************************"
wget -P dataset/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget -P dataset/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget -P dataset/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip

echo "********************************** Unzipping Questions... **********************************"
unzip -q dataset/v2_Questions_Train_mscoco.zip -d dataset/
unzip -q dataset/v2_Questions_Test_mscoco.zip -d dataset/
unzip -q dataset/v2_Questions_Val_mscoco.zip -d dataset/

echo "********************************** Downloading Images... **********************************"
wget -P dataset/ http://images.cocodataset.org/zips/train2014.zip
wget -P dataset/ http://images.cocodataset.org/zips/val2014.zip
wget -P dataset/ http://images.cocodataset.org/zips/test2015.zip

echo "********************************** Unzipping Images... **********************************"
unzip -q dataset/train2014.zip -d dataset/
unzip -q dataset/val2014.zip -d dataset/
unzip -q dataset/test2015.zip -d dataset/

echo "Downloading ********************************** Complementary Pairs... **********************************"
wget -P dataset/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip
wget -P dataset/ https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip

echo "Unzipping ********************************** Complementary Pairs... **********************************"
unzip -q dataset/v2_Complementary_Pairs_Train_mscoco.zip -d dataset/
unzip -q dataset/v2_Complementary_Pairs_Val_mscoco.zip -d dataset/