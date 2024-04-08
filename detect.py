import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import json
import pandas as pd
from fake_celeb_dataset import FakeAVceleb
from model import MP_AViT, MP_av_feature_AViT
from subprocess import call
from backbone.select_backbone import select_backbone
from torch.optim import Adam
from config_deepfake import load_opts, save_opts
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data.sampler import RandomSampler, Sampler
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Optional, Union
from audio_process import AudioEncoder
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import math
import h5py
import os
import glob
import time
from deep_fake_data import prepocess_video
import logging
from load_audio import wav2filterbanks, wave2input
from torch.utils.tensorboard import SummaryWriter
from transformer_component import transformer_decoder
import pickle
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger
from tqdm.contrib.logging import logging_redirect_tqdm

opts = load_opts()
device = opts.device
#local_rank = opts.local_rank
#torch.cuda.set_device()
#torch.distributed.init_process_group(backend='nccl')
device = torch.device(device)

with open('pca.pkl', 'rb') as pickle_file:
    pca = pickle.load(pickle_file)

def get_logger(filename, verbosity=1, name=__name__):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # sh = logging.StreamHandler()
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)

    return logger

# class Testset_new(Dataset):
#     def __init__(self, h5_file, fake_type, max_len, data_list):
#         super(Testset_new, self).__init__()
#         self.h5_file = h5_file
#         self.max_len = max_len
#         self.fake_type = fake_type
#         self.data_list = data_list
#     def __len__(self):
#         with h5py.File(self.h5_file, 'r') as f:
#             return len(f[self.fake_type])
#     def __getitem__(self, index):
#         with h5py.File(self.h5_file, 'r') as f:
#             data_total = f[self.fake_type][str(index)][:, :]
#             data_total = torch.from_numpy(data_total)
#             data = data_total
#             mask = torch.zeros(self.max_len)
#             if data.shape[0] < self.max_len:
#                 pad_len = self.max_len - data.shape[0]
#                 mask[data.shape[0]:] = 1.0
#                 data = F.pad(data, (0, 0, 0, pad_len))
#             else:
#                 start = 0
#                 data = data[start:start+self.max_len, :]
#                 pad_len = 0
#         return (data, mask, pad_len)

class network(nn.Module):
    def __init__(self, vis_enc, aud_enc, transformer):
        super().__init__()
        self.vis_enc = vis_enc
        self.aud_enc = aud_enc
        self.transformer = transformer

    def forward(self, video, audio, phase=0, train=True):
        if train:
            if phase == 0:
                vid_emb = self.vis_enc(video)
                batch_size,c,t, h, w= vid_emb.shape
                vid_emb = vid_emb[:, None]
                vid_emb = vid_emb.expand(-1, batch_size, -1, -1, -1, -1)
                vid_emb = vid_emb.reshape(-1, c, t, h, w)
                aud_emb = self.aud_enc(audio)
                batch_size, c_aud, t_aud = aud_emb.shape
                aud_emb = aud_emb[None, :]
                aud_emb = aud_emb.expand(batch_size, -1, -1, -1).reshape(-1, c_aud, t_aud)
                cls_emb = self.transformer(vid_emb, aud_emb)
            elif phase == 1:
                vid_emb = self.vis_enc(video)
                batch_size,c,t, h, w= vid_emb.shape
                vid_emb = vid_emb[:, None]
                vid_emb = vid_emb.expand(-1, opts.number_sample, -1, -1, -1, -1)
                vid_emb = vid_emb.reshape(-1, c, t, h, w)
                aud_emb = self.aud_enc(audio)
                batch_size, c_aud, t_aud = aud_emb.shape
                aud_emb_new = torch.zeros_like(aud_emb)
                aud_emb_new = aud_emb_new[None, :]
                aud_emb_new = aud_emb_new.expand(opts.number_sample, -1, -1, -1).reshape(-1, c_aud, t_aud)
                num_sample = opts.number_sample
                if batch_size == num_sample*(opts.bs2):
                    for k in range(opts.bs2):
                        aud_emb_new[k*num_sample*num_sample:(k+1)*num_sample*num_sample] = (aud_emb[k*num_sample:(k+1)*num_sample][None, :]).expand(opts.number_sample, -1, -1, -1).reshape(-1, c_aud, t_aud)
                else:
                    bs2 = int(batch_size / num_sample)
                    assert batch_size == bs2 * num_sample
                    for k in range(bs2):
                        aud_emb_new[k*num_sample*num_sample:(k+1)*num_sample*num_sample] = (aud_emb[k*num_sample:(k+1)*num_sample][None, :]).expand(opts.number_sample, -1, -1, -1).reshape(-1, c_aud, t_aud)
                aud_emb = aud_emb_new
                cls_emb = self.transformer(vid_emb, aud_emb)
            
        else:
            vid_emb = self.vis_enc(video)
            aud_emb = self.aud_enc(audio)
            cls_emb = self.transformer(vid_emb, aud_emb)
        return cls_emb

def test2(dist_model, avfeature_model, loader, dist_reg_model, avfeature_reg_model, max_len=50):
    output_dir = opts.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger_path = os.path.join(output_dir, 'output.log')
    score_file_path = os.path.join(output_dir, 'testing_scores.npy')
    logger = get_logger(logger_path)
    logger.info('Start testing!')
    score_list = []
    with logging_redirect_tqdm():
        with tqdm(total=len(loader), position=0, leave=False, colour='green', ncols=150) as pbar:
            for nm, aud_vis in enumerate(loader):
                video_set = aud_vis['video']
                audio_set = aud_vis['audio']
                path_for_detect = aud_vis['sample']
                #print(path_for_detect)
                pbar.set_postfix(data_path = path_for_detect)
                time_len = video_set.shape[2]
                #predict_set = np.zeros((time_len - 5 + 1, 31))
                if (time_len -5 +1) < max_len:
                    max_seq_len = time_len - 5 + 1
                else:
                    max_seq_len = max_len
                predict_set = np.zeros((max_seq_len, 31))
                predict_set_avfeature = np.zeros((max_seq_len, 31))
                #real_result.append(1)
                #fake_result.append(1)
                #for k in tqdm(range(time_len - 5 + 1)):
                for k in tqdm(range(max_seq_len), position=1, leave=False, colour='red',ncols=80):
                    #video_set = torch.permute(video_set, (0, 2, 1, 3, 4))
                    video = video_set[:, :, k:k+5, :, :]
                    #video = video / 255.0
                    audio = audio_set[:, (k+15-15)*opts.aud_fact:(k+5+15+15)*opts.aud_fact]
                    '''
                    audio = aud_vis['audio'].to(device)
                    audio, _, _, _ = wav2filterbanks(audio)
                    audio = audio.permute([0, 2, 1])[:, None]
                    video = aud_vis['video'].to(device)
                    '''
                    dist_model.eval()
                    avfeature_model.eval()
                    with torch.no_grad():
                        batch_size = video.shape[0]
                        b, c ,t, h, w = video.shape
                        video = video[:, None]
                        video = video.repeat(1,31 , 1, 1, 1, 1).reshape(-1, c, t, h, w).to(device)
                        audio_list = []
                        for j in range(batch_size):
                            for i in range(31):
                                audio_list.append(audio[j:j+1, i*opts.aud_fact:(i+5)*opts.aud_fact])
                        audio = torch.cat(audio_list, dim=0).to(device)
                        #audio = wave2input(audio, device=device)
                        audio, _, _, _ = wav2filterbanks(audio.to(device), device=device)
                        audio = audio.permute([0, 2, 1])[:, None]
                        score = dist_model(video, audio, train=False)
                        avfeature = avfeature_model(video, audio, train=False)[15]
                        score = score.reshape(batch_size, 31)
                        avfeature = avfeature.cpu().numpy()[None, ...]
                        avfeature = pca.transform(avfeature)
                        #predict = torch.argmax(score, dim=1)
                        #distribution[0, predict.item()] += 1
                        #real_result[-1] = real_result[-1] * real_distribution[0, predict.item()]
                        #fake_result[-1] = fake_result[-1] * fake_distribution[0, predict.item()]
                        #predict = torch.abs(predict - 15)
                        #predict_set.append(predict.item())
                        predict_set[k] = score.squeeze(0).cpu().numpy()
                        predict_set_avfeature[k] = avfeature
                    #print('--------------------computing score for video-------------------')
                dist_reg_model.eval()
                avfeature_reg_model.eval()
                mask = torch.zeros(max_len)
                predict_set = torch.from_numpy(predict_set)
                predict_set_avfeature = torch.from_numpy(predict_set_avfeature)
                criterion = nn.KLDivLoss(reduce=False)
                criterion_av_feature = nn.MSELoss(reduction="none")
                if predict_set.shape[0] < max_len:
                    pad_len = max_len - predict_set.shape[0]
                    mask[predict_set.shape[0]:] = 1.0
                    seq = F.pad(predict_set, (0, 0, 0, pad_len))
                    seq_avfeature = F.pad(predict_set_avfeature, (0, 0, 0, pad_len))
                else:
                    start = 0
                    seq = predict_set[start:start+max_len, :]
                    seq_avfeature = predict_set_avfeature[start:start+max_len, :]
                    pad_len = 0
                seq = seq[None, :, :]
                seq_avfeature = seq_avfeature[None, :, :]
                seq = seq.to(device)
                seq_avfeature = seq_avfeature.to(device)
                mask = mask.to(device)
                mask = mask[None, :]
                with torch.no_grad():
                    target = seq[:, 1:, :]
                    target = nn.functional.softmax(target.float(), dim=2)
                    target_av_feature = seq_avfeature[:, 1:, :]
                    target_av_feature = F.normalize(target_av_feature, p=2.0, dim=2)
                    input = seq[:, :-1, :]
                    input_av_feature = seq_avfeature[:, :-1, :]
                    input_av_feature = F.normalize(input_av_feature, p=2.0, dim=2)
                    input_mask_ = mask[:, :-1]
                    logit= dist_reg_model(input.float(), input_mask_)
                    logit_avfeature = avfeature_reg_model(input_av_feature.float(), input_mask_)
                    logit = nn.functional.log_softmax(logit, dim=2)
                    prob_total = criterion(logit, target)
                    prob_total_avfeature = criterion_av_feature(logit_avfeature, target_av_feature)
                    prob = prob_total[0, :(max_len - pad_len -1)]
                    prob_avfeature = prob_total_avfeature[0, :(max_len - pad_len -1)]
                    prob = torch.sum(prob, dim=1)
                    prob = torch.mean(prob)
                    prob_avfeature = torch.sum(prob_avfeature, dim=1)
                    prob_avfeature = torch.mean(prob_avfeature)
                    prob = (opts.lam)*prob_avfeature + prob
                #tqdm.write("The score of this video is {} ".format(prob.item()))
                logger.info("The score of this video is {} ".format(prob.item()))
                score_list.append(prob.item())
                pbar.update(1)
            np.save(score_file_path, np.array(score_list))
            logger.info('Finished!')
            


def main():
    fake_distribution = np.zeros(31)
    vis_enc, _ = select_backbone(network='r18')
    aud_enc = AudioEncoder()
    #lrs2_distribution = np.zeros(31)
    Transformer = MP_AViT(image_size=14, patch_size=0, num_classes=1, dim=512, depth=3, heads=4, mlp_dim=512,  dim_head=128, dropout=0.1, emb_dropout=0.1, max_visual_len=5, max_audio_len=4)
    avfeature_Transformer = MP_av_feature_AViT(image_size=14, patch_size=0, num_classes=1, dim=512, depth=3, heads=4, mlp_dim=512,  dim_head=128, dropout=0.1, emb_dropout=0.1, max_visual_len=5, max_audio_len=4)
    sync_model = network(vis_enc=vis_enc, aud_enc=aud_enc, transformer=Transformer)
    avfeature_sync_model = network(vis_enc=vis_enc, aud_enc=aud_enc, transformer=avfeature_Transformer)
    #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    sync_model_weight = torch.load('sync_model.pth', map_location=torch.device('cpu'))
    sync_model.load_state_dict(sync_model_weight)
    avfeature_sync_model.load_state_dict(sync_model_weight)
    sync_model.to(device)
    avfeature_sync_model.to(device)
    #model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    dist_regressive_model = transformer_decoder(input_dim_old=31, input_dim=256, compress_factor=1, num_heads=16, dropout_prob=0.1, max_len=49, layers=2)
    avfeature_regressive_model = transformer_decoder(input_dim_old=31, input_dim=256, compress_factor=1, num_heads=16, dropout_prob=0.1, max_len=49, layers=2)
    reg_model_weight = torch.load('dist_regressive_model.pth', map_location=torch.device("cpu"))
    avfeature_reg_model_weight = torch.load('avfeature_regressive_model.pth', map_location=torch.device("cpu"))
    dist_regressive_model.load_state_dict(reg_model_weight)
    dist_regressive_model.to(device)
    avfeature_regressive_model.load_state_dict(avfeature_reg_model_weight)
    avfeature_regressive_model.to(device)
    if opts.test_video_path is not None:
        if opts.test_video_path.split('.')[-1] == 'mp4':
            test_video = FakeAVceleb([opts.test_video_path], opts.resize, opts.fps, opts.sample_rate, vid_len=opts.vid_len, phase=0, train=False, number_sample=1, lrs2=False, need_shift=False, lrs3=False, kodf=False, lavdf=False, robustness=False, test=True)
        else:
            with open(opts.test_video_path) as file:
                test_video_path = file.readlines()
            test_video = FakeAVceleb(test_video_path, opts.resize, opts.fps, opts.sample_rate, vid_len=opts.vid_len, phase=0, train=False, number_sample=1, lrs2=False, need_shift=False, lrs3=False, kodf=False, lavdf=False, robustness=False, test=True)
    loader_test = DataLoader(test_video, batch_size=opts.bs, num_workers=opts.n_workers, shuffle=False)
    test2(sync_model, avfeature_sync_model, loader_test, dist_reg_model=dist_regressive_model, avfeature_reg_model=avfeature_regressive_model, max_len=opts.max_len)



if __name__ == '__main__':
    main()