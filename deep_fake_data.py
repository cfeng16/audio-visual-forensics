import os
import pickle
import threading
import glob
import numpy as np
from torch.utils import data
from tqdm import tqdm
from torchvision.transforms import InterpolationMode
from load_audio import load_wav
from load_video import load_mp4
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from shutil import rmtree
import subprocess
from PIL import Image

def prepocess_video(videofile, opts, resize, reference):
    if os.path.exists(os.path.join(opts.tmp_dir,reference)):
        rmtree(os.path.join(opts.tmp_dir,reference))

    os.makedirs(os.path.join(opts.tmp_dir,reference))

    command = ("ffmpeg -y -i %s -threads 1 -f image2 %s" % (videofile,os.path.join(opts.tmp_dir,reference,'%06d.jpg'))) 
    output = subprocess.call(command, shell=True, stdout=None)

    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (videofile,os.path.join(opts.tmp_dir,reference,'audio.wav'))) 
    output = subprocess.call(command, shell=True, stdout=None)
    
    # -- load video
    #vid_path_orig = os.path.join(self.data_path, vid_name + '.mp4')
    #vid_path_25fps = os.path.join(self.data_path, vid_name + '.mp4')

    images = []
        
    flist = glob.glob(os.path.join(opts.tmp_dir,reference,'*.jpg'))
    flist.sort()

    for fname in flist:
        images.append(np.array(Image.open(fname)))
    video = np.stack(images,axis=0)
    #im = np.expand_dims(im,axis=0)
    #video = np.transpose(im,(0,3,4,1,2))
    video = __load_video__(video, resize=resize)
    aud_path = os.path.join(opts.tmp_dir,reference,'audio.wav')
    audio = load_wav(aud_path).astype('float32')
    vid_time_len = video.shape[1]
    fps = opts.fps  # TODO: get as param?
    aud_fact = int(np.round(opts.sample_rate / fps))
    audio, video = trunkate_audio_and_video(video, audio, aud_fact)
    assert aud_fact * video.shape[0] == audio.shape[0]
    video = video.transpose([3, 0, 1, 2])
    #start = np.random.randint(low=0, high=vid_time_len-self.vid_len+1)
    '''
    start = np.random.randint(low=0, high=(vid_time_len-opts.vid_len+1))
    video = video[:, start:start+opts.vid_len, :, :]
    audio = audio[start*aud_fact : (start+opts.vid_len)*aud_fact]
    '''
    assert aud_fact * video.shape[1] == audio.shape[0]
    #audio = np.pad(audio, (15*aud_fact, 15*aud_fact), 'constant', constant_values=(0,0))
    #vid_time_len = video.shape[1]
    #start = np.random.randint(low=0, high=(vid_time_len-opts.vid_len+1))
    #video = video[:, start:start+opts.vid_len, :, :]
    #start_audio = (start-15 + 15)*aud_fact
    #end_audio = (start+opts.vid_len+15 + 15)*aud_fact
    #audio = audio[start_audio : end_audio]
    #audio = np.array(audio)
    audio = np.expand_dims(audio,axis=0)
    video = np.expand_dims(video,axis=0)
    out_dict = {
            'video': video,
            'audio': audio,
        }
    return out_dict


def __load_video__(frames, resize=None):

        #frames = load_mp4(vid_path)

        if resize:
            import torchvision
            from PIL import Image
            ims = [Image.fromarray(frm) for frm in frames]
            ims = [
                torchvision.transforms.functional.resize(im,
                                                         resize,
                                                         interpolation=InterpolationMode.BICUBIC)
                for im in ims
            ]
            frames = np.array([np.array(im) for im in ims])

        return frames.astype('float32')


def trunkate_audio_and_video(video, aud_feats, aud_fact):

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