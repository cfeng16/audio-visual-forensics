Self-Supervised Video Forensics by Audio-Visual Anomaly Detection
==================================================================
**Chao Feng, Ziyang Chen, Andrew Owens**  
**University of Michigan, Ann Arbor**

**CVPR 2023 (Highlight)**

---

<div>✅ Forensics auto regressive model</div>
<div>✅ Audio-visual synchronization model</div> 
<div>[ ] Detection File</div>

---

Visual encoder code is in folder backbone, audio encoder code is in audio_process.py, and audio-visual synchronization transformer code is av_sync_model.py

Audio-visual synchronization model and Forensics autoregressive model checkpoint [Google Drive](https://drive.google.com/drive/folders/1Mqbjlyk3R7Ba8pktsYXVqt0kIdQ_SgMT?usp=drive_link)

Audio-visual synchronization model code is based on [vit-pytorch](https://github.com/lucidrains/vit-pytorch)

Decoder only autoregressive model is partially based on [memory-compressed-attention](https://github.com/lucidrains/memory-compressed-attention)

Visual encoder is heavily borrowed from [action classifiction](https://github.com/TengdaHan/ActionClassification)