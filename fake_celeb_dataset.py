import os
import pickle
import threading

import numpy as np
from torch.utils import data
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from load_audio import load_wav
from load_video import load_mp4
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class FakeAVceleb(data.Dataset):

    def __init__(self, video_path, resize, fps, sample_rate, vid_len, phase, train=True, number_sample=1, lrs2=False, need_shift=False, lrs3=False, kodf=False, real=True, lavdf=False, vox_korea=False, random_shift=False, fixed_shift=False, shift=0,robustness=False, test=False):
        super(FakeAVceleb, self).__init__()
        self.resize = resize
        self.fps = fps
        self.sample_rate = sample_rate
        self.vid_len = vid_len
        self.train = train
        self.lrs2 = lrs2
        self.lrs3 = lrs3
        self.real = real
        self.kodf = kodf
        self.lavdf = lavdf
        self.random_shift = random_shift
        self.fixed_shift = fixed_shift
        self.shift = shift
        self.vox_korea = vox_korea
        self.robustness = robustness
        self.test = test
        if self.lrs2:
            self.data_path = '/datab/chfeng/mvlrs_v1/pretrain'
        elif self.lrs3:
            self.data_path = '/datab/chfeng/lrs3'
        elif self.kodf:
            if self.real:
                self.data_path = '/datab/chfeng/syncnet_python/real_set/pycrop'
            else:
                self.data_path =  '/datab/chfeng/syncnet_python/fake_set/pycrop'
        elif self.lavdf:
            self.data_path = '/datab/chfeng/lavdf'
        elif self.vox_korea:
            self.data_path = '/datad/chfeng/vox_korea'
        elif self.robustness:
            self.data_path = '/datad/chfeng/DeeperForensics-1.0'
        elif self.test:
            self.data_path = ''
        else:
            self.data_path = '/datab/chfeng/av_sync'
        self.phase = phase
        self.all_vids = video_path
        self.number_sample = number_sample
        self.need_shift = need_shift
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.all_vids)

    def __getitem__(self, index):
        'Generates one sample of data'
        assert index < self.__len__()

        vid_path = self.all_vids[index]
        if self.lrs2:
            if self.train:
                vid_path = vid_path.split('\n')[0]
            else:
                vid_path = vid_path.split(' ')[0]
        elif self.lrs3:
            vid_path = vid_path.split('.')[0]
        elif self.kodf:
            vid_path = vid_path
        elif self.lavdf:
            vid_path = vid_path
        elif self.vox_korea:
            vid_path = vid_path.split('.')[0]
        elif self.robustness:
            vid_path = vid_path.split('.')[0]
        elif self.test:
            vid_path = vid_path.split('.')[0]
        else:
            vid_path = vid_path.split('.')[0]
            vid_path = vid_path.split(' ')
            if len(vid_path) == 1:
                vid_path = vid_path[0]
            elif len(vid_path) == 2:
                vid_path = vid_path[0] + vid_path[1]
            else:
                raise Exception('That is impossible')
        #vid_name, vid_ext = os.path.splitext(vid_path)
        if self.train:
            vid_name = vid_path
        else:
            vid_name = vid_path

        # -- load video
        if self.kodf:
            vid_path_orig = os.path.join(self.data_path, vid_name + '.avi')
            vid_path_25fps = os.path.join(self.data_path, vid_name + '.mp4')
        else:
            vid_path_orig = os.path.join(self.data_path, vid_name + '.mp4')
            vid_path_25fps = os.path.join(self.data_path, vid_name + '.mp4')
        # -- reencode video to 25 fps
        
        command = (
            "ffmpeg -threads 1 -loglevel error -y -i {} -an -r 25 {}".format(
                vid_path_orig, vid_path_25fps))
        from subprocess import call
        
        cmd = command.split(' ')
        #print('Resampling {} to 25 fps'.format(vid_path_orig))
        #call(cmd)

        video = self.__load_video__(vid_path_25fps, resize=self.resize)

        aud_path = os.path.join(self.data_path, vid_name + '.wav')
        if not os.path.exists(aud_path):  # -- extract wav from mp4
            command = (
                ("ffmpeg -threads 1 -loglevel error -y -i {} "
                    "-async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {}")
                .format(vid_path_orig, aud_path))
            from subprocess import call
            cmd = command.split(' ')
            call(cmd)

        audio = load_wav(aud_path).astype('float32')

        fps = self.fps  # TODO: get as param?
        aud_fact = int(np.round(self.sample_rate / fps))
        audio, video = self.trunkate_audio_and_video(video, audio, aud_fact)
        assert aud_fact * video.shape[0] == audio.shape[0]
        audio = np.array(audio)
        #video = video[0:30, :, :, :]
        #audio = audio[0:(30*aud_fact)]
        if self.need_shift:
            if self.random_shift:
                shift = np.random.randint(-15, 15, 1)
            elif self.fixed_shift:
                shift = self.shift
        else:
            shift = np.array([0])
        true_shift = shift
        audio_len = audio.shape[0]
        '''
        if shift == 0:
            audio = audio
        elif shift < 0:
            audio[:(audio_len - shift*aud_fact)] = audio[shift*aud_fact:]
            audio[(audio_len - shift*aud_fact):] = 0
        elif shift > 0:
            audio[shift*aud_fact:] = audio[:(audio_len - shift*aud_fact)]
            audio[:shift*aud_fact] = 0
        '''
        assert aud_fact * video.shape[0] == audio.shape[0]
        video = video.transpose([3, 0, 1, 2])  # t h w c -> c t h w
        if shift[0] == 0:
            audio = np.pad(audio, (15*aud_fact, 15*aud_fact), 'constant', constant_values=(0,0))
        elif shift[0] > 0:
            shift = np.abs(shift[0])
            audio = np.pad(audio, ((15 + shift)*aud_fact, (15 - shift)*aud_fact), 'constant', constant_values=(0,0))
        elif shift[0] < 0:
            shift = np.abs(shift[0])
            audio = np.pad(audio, ((15 - shift)*aud_fact, (15 + shift)*aud_fact), 'constant', constant_values=(0,0))
        #audio = np.expand_dims(audio,axis=0)
        #video = np.expand_dims(video,axis=0)
        
        out_dict = {
            'video': video,
            'audio': audio,
            'sample': vid_path,
            'shift':true_shift
        }

        return out_dict

    def __load_video__(self, vid_path, resize=None):

        frames = load_mp4(vid_path)

        if resize:
            import torchvision
            from PIL import Image
            ims = [Image.fromarray(frm) for frm in frames]
            ims = [
                torchvision.transforms.functional.resize(im,
                                                         [resize, resize], 
                                                         interpolation=InterpolationMode.BICUBIC)
                for im in ims
            ]
            frames = np.array([np.array(im) for im in ims])

        return frames.astype('float32')

    def trunkate_audio_and_video(self, video, aud_feats, aud_fact):

        aud_in_frames = aud_feats.shape[0] // aud_fact

        # make audio exactly devisible by video frames
        aud_cutoff = min(video.shape[0], int(aud_feats.shape[0] / aud_fact))

        aud_feats = aud_feats[:aud_cutoff * aud_fact]
        aud_in_frames = aud_feats.shape[0] // aud_fact

        min_len = min(aud_in_frames, video.shape[0])

        # --- trunkate all to min
        video = video[:min_len]
        aud_feats = aud_feats[:min_len * aud_fact]
        if not aud_feats.shape[0] // aud_fact == video.shape[0]:
            import ipdb
            ipdb.set_trace(context=20)

        return aud_feats, video
