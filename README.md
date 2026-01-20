# Beyond Shadows: A Large-Scale Benchmark for High-Fidelity Facial Shadow Removal

<div>
<span class="author-block">
  Tailong Luo
</span>,
  <span class="author-block">
    Yihang Dong
  </span>,
  <span class="author-block">
    <a href='https://baijiesong.github.io/'>Jiesong Bai</a>
  </span>,
  <span class="author-block">
    Junyu Xia
  </span>,
  <span class="author-block">
    Jinyang Huang
  </span>,
  <span class="author-block">
    Wangyu Wu
  </span>,
  <span class="author-block">
    <a href='https://cxh.netlify.app/'>Xuhang Chen</a><sup> 📮</sup>
  </span>
  (📮 Corresponding author)
</div>

<b>University of Macau, Shanghai Jiao Tong University, SIAT CAS, Xi'an Jiaotong-Liverpool University, Huizhou Univeristy</b>

In <b>_IEEE International Conference on Acoustics, Speech, and Signal Processing 2026 (ICASSP 2026)_</b>

# 🔮 Dataset

The benchmark datasets are available at [Kaggle](https://www.kaggle.com/datasets/xuhangc/facialshadowremoval).

# ⚙️ Usage

## Training
You may download the dataset first, and then specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in `config.yml`.

For single GPU training:
```
python train.py
```
For multiple GPUs training:
```
accelerate config
accelerate launch train.py
```
If you have difficulties with the usage of `accelerate`, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

## Inference

Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in `config.yml`.

```bash
python test.py
```
