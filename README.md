# 🧬 m6A-HGB-GHOST Inference Pipeline (Docker Edition)

This repository contains the **Dockerized inference pipeline** for our HGB-GHOST model, designed to predict m6A modification sites across SG-NEx and other RNA datasets.

---

## 📦 Overview

- The **Docker image** is available under this repository’s **Packages** section:
  ```
  ghcr.io/jingzhing/m6a-hgb-ghost-infer:v2
  ```

- Your Ronin instance or any compatible Ubuntu cloud VM should have two directories (Creating Below):
  ```
  ~/data_in/     # input JSON or JSON.GZ datasets
  ~/data_out/    # model prediction outputs
  ```
- To test, you can just use the any dataset given for the project.


## 🚀 How to Run with Docker on RONIN (Start a Docker Machine instead of an utunbu one)

Video Tutorial: https://drive.google.com/file/d/1PoQcxuXj5sF3mkNqJbSR-9Odzx2qKJk6/view?usp=sharing

### 1. Create & then Connect to the Ronin (Docker!!!) Instance & create data folders
```bash
ssh -i "/xxx/key.pem" ubuntu@<RONIN_IP>
```
```bash
mkdir -p ~/data_in ~/data_out
```

### 2. Pull or Update the Docker Image
```bash
docker pull ghcr.io/jingzhing/m6a-hgb-ghost-infer:v2
```

### 3. Upload Your Dataset to the Cloud
From your **local machine**, copy any dataset to the remote instance:

```bash
scp -i "/xxx/key.pem" "xxxx\xxxx\dataset2.json.gz" ubuntu@<RONIN_IP>:data_in/
```
---

### 4. Run Inference on a Dataset
Run Docker with mounted directories for input and output:
(If you are using another dataset, change the filenames below of dataset2.json.gz to your data file)
```bash
docker run --rm \
  -v ~/data_in:/data_in \
  -v ~/data_out:/data_out \
  ghcr.io/jingzhing/m6a-hgb-ghost-infer:v2 \
  predict \
  --json /data_in/dataset2.json.gz \
  --model /opt/model/model_tuned.joblib \
  --output /data_out/dataset2_preds.csv \
  --emit labels
```

The output file will appear in:
```
~/data_out/preds2.csv
```

## 📤 Copy All Outputs Back to Local

From your local terminal:
```bash
scp -i "/xxx/key.pem" ubuntu@<RONIN_IP>:~/data_out/*.csv "xxx/Downloads/"
```

**Maintainer:**  
👤 *exonintron group*  


