Self-Supervised Video Forensics by Audio-Visual Anomaly Detection
==================================================================
**Chao Feng, Ziyang Chen, Andrew Owens**  
**University of Michigan, Ann Arbor**

**CVPR 2023 (Highlight)**

---

This is the code for audio-visual forensics.

Steps to run the python code directly:

`pip install -r requirements.txt`

```python
# 1. test a sample fake video (path of video should be full path)
CUDA_VISIBLE_DEVICES=8 python detect.py --test_video_path /home/xxxx/test_video.mp4 --device cuda:0 --max-len 50 --n_workers 4  --bs 1 --lam 0 --output_dir /home/xxx/save 
# 2. test a list of fake videos (path of .txt file should be full path, and list should contain full paths of testing videos)
CUDA_VISIBLE_DEVICES=8 python detect.py --test_video_path /home/xxxx/fake_videos.txt --device cuda:0 --max-len 50 --n_workers 4 --bs 1 --lam 0 --output_dir /home/xxx/save
```

(lam is a hyperparameter you can tune to combine scores from distributions over delays and audio-visual network activations mentioned in [paper](https://arxiv.org/pdf/2301.01767.pdf) method section. Default lam=0 is distributions over delays only.)

Audio-visual synchronization model checkpoint `sync_model.pth` can be donwloaded by this [link](https://drive.google.com/file/d/1BxaPiZmpiOJDsbbq8ZIDHJU7--RJE7Br/view?usp=sharing). Noted that AV synchronization model consists of video branch, audio branch, and audio-visual feature fusion transformer.

In the end, there would be a `output.log` file and a `testing_score.npy` file under output_dir generated to record scores for all the testing videos.

---

Audio-visual synchronization model code is based on [vit-pytorch](https://github.com/lucidrains/vit-pytorch)

Decoder only autoregressive model is partially based on [memory-compressed-attention](https://github.com/lucidrains/memory-compressed-attention)

Visual encoder is heavily borrowed from [action classifiction](https://github.com/TengdaHan/ActionClassification)

---

Any questions please contact chfeng@umich.edu, I will try to respond ASAP, sorry for any delay before. 

---

```text
@inproceedings{feng2023self,
  title={Self-supervised video forensics by audio-visual anomaly detection},
  author={Feng, Chao and Chen, Ziyang and Owens, Andrew},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10491--10503},
  year={2023}
}
```