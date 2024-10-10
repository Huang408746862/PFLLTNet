# PixFusion-LLTNet: Enhancing Low-Light Images with PixelShuffle Upsampling and Feature Fusion

## Description
This is the TensorFlow version of PixFusion-LLTNet.

## Experiment

### 1. Create Environment
- Make Conda Environment
```bash
conda create -n PFLLTNet python=3.10
conda activate PFLLTNet
```
- Install Dependencies
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
pip install tensorflow==2.10 opencv-python numpy tqdm matplotlib lpips
```

### 2. Prepare Datasets
Download the LOLv1 and LOLv2 datasets:

LOLv1 - [Google Drive](https://drive.google.com/file/d/1vhJg75hIpYvsmryyaxdygAWeHuiY_HWu/view?usp=sharing)

LOLv2 - [Google Drive](https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view?usp=sharing)

**Note:** Under the main directory, create a folder called ```data``` and place the dataset folders inside it.
<details>
  <summary>
  <b>Datasets should be organized as follows:</b>
  </summary>

    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |     ...
    |    |--LOLv2
    |    |    |--Real_captured
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |     ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |     ...
    |    |    |--Synthetic
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |    ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |    ...

</details>

**Note:** ```data``` directory should be placed under the ```TensorFlow``` implementation folder.

### 3. Test
You can test the model using the following commands.  GT Mean evaluation can be done with the ```--gtmean``` argument.

```bash
# Test on LOLv1
python main.py --test --dataset LOLv1 --weights pretrained_weights/LOLv1.h5
# Test on LOLv1 using GT Mean
python main.py --test --dataset LOLv1 --weights pretrained_weights/LOLv1.h5 --gtmean

# Test on LOLv2 Real
python main.py --test --dataset LOLv2_Real --weights pretrained_weights/LOLv2_Real.h5
# Test on LOLv2 Real using GT Mean
python main.py --test --dataset LOLv2_Real --weights pretrained_weights/LOLv2_Real.h5 --gtmean

# Test on LOLv2 Synthetic
python main.py --test --dataset LOLv2_Synthetic --weights pretrained_weights/LOLv2_Synthetic.h5
# Test on LOLv2 Synthetic using GT Mean
python main.py --test --dataset LOLv2_Synthetic --weights pretrained_weights/LOLv2_Synthetic.h5 --gtmean
```

### 4. Compute Complexity
You can test the model complexity (FLOPS/Params) using the following command:
```bash
# To run FLOPS check with default (1,256,256,3)
python main.py --complexity

# To run FLOPS check with custom (1,H,W,C)
python main.py --complexity --shape '(H,W,C)'
```

### 5. Train
You can train the model using the following commands:

```bash
# Train on LOLv1
python main.py --train --dataset LOLv1

# Train on LOLv2 Real
python main.py --train --dataset LOLv2_Real

# Train on LOLv2 Synthetic
python main.py --train --dataset LOLv2_Synthetic
```
## Results

|          | Original    |         |       |GTMean    |        |        |
|----------|-------------|---------|-------|----------|--------|--------|
|Dataset   |PSNR         |SSIM     |LPIPS  |PSNR      | SSIM   |LPIPS   |
| LOLv1    | 23.44       | 0.848   | 0.068 |26.81     | 0.863  | 0.065  |
| LOLv2-R  | 26.49       | 0.884   | 0.053 |29.91     | 0.894  | 0.051  |
| LOLv2-S  | 23.73       | 0.922   | 0.044 |29.82     | 0.943  | 0.032  |

## Citation

