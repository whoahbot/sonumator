from IPython.display import Audio
import mimetypes
import torchaudio
from fastai.data_block import ItemBase
from fastai.vision import Image
import numpy as np
import torch
import warnings
from pathlib import Path, PosixPath


AUDIO_EXTENSIONS = tuple(str.lower(k) for k, v in mimetypes.types_map.items() if v.startswith('audio/'))

class AudioItem(ItemBase):
    def __init__(self, sig=None, sr=None, path=None, spectro=None, max_to_pad=None, start=None, end=None):
        '''Holds Audio signal and/or specrogram data'''
        if isinstance(sig, np.ndarray): sig = torch.from_numpy(sig)
        self._sig, self._sr, self.path, self.spectro = sig, sr, path, spectro
        self.max_to_pad = max_to_pad
        self.start, self.end = start, end

    def __str__(self):
        return f'{self.__class__.__name__} {round(self.duration, 2)} seconds ({self.nchannels} channels, {self.nsamples} samples @ {self.sr}hz)'

    def __len__(self): return self.data.shape[0]

    def _repr_html_(self):
        return f'{self.__str__()}<br />{self.ipy_audio._repr_html_()}'

    def reconstruct(self, t): return(AudioItem(spectro=t))

    def show(self, title: [str] = None, **kwargs):
        print(f"File: {self.path}")
        print(f"Total Length: {round(self.duration, 2)} seconds")
        print(f"Number of Channels: {self.nchannels}")
        images_per_channel = len(self.get_spec_images())/self.nchannels
        self.hear(title=title)
        for i,im in enumerate(self.get_spec_images()):
            print(f"Channel {int(i//images_per_channel)}.{int(i%images_per_channel)} ({im.shape[-2]}x{im.shape[-1]}):")
            display(im.rotate(180).flip_lr())
            
                         
    def get_spec_images(self):
        sg = self.spectro
        if sg is None: return [] 
        return [Image(s.unsqueeze(0)) for s in sg]

    def hear(self, title=None):
        if title is not None: print("Label:", title)
        if self.sig is None: self._check_signal()
        if self.start is not None or self.end is not None:
            print(f"{round(self.start/self.sr, 2)}s-{round(self.end/self.sr,2)}s of original clip")
            start = 0 if self.start is None else self.start
            end = self.nsamples-1 if self.end is None else self.end
            display(Audio(data=self.sig[:,start:end], rate=self.sr))
        else:
            display(self.ipy_audio)
        
    
    def apply_tfms(self, tfms):
        for tfm in tfms:
            self.data = tfm(self.data)
        return self

    def _reload_signal(self): self._sig,self._sr = torchaudio.load(self.path)

    @property
    def sig(self):
        if self._sig is None: self._reload_signal()
        return self._sig
    @sig.setter
    def sig(self, sig): self._sig = sig

    @property
    def sr(self):
        if self._sr is None: self._reload_signal()
        return self._sr
    @sr.setter
    def sr(self, sr): self._sr=sr

    @property
    def shape(self): return self.data.shape

    @property
    def ipy_audio(self): 
        if self.sig is None: self._check_signal()
        return Audio(data=self.sig, rate=self.sr)

    @property
    def duration(self): 
        if(self.sig is not None): return self.nsamples/self.sr
        else: 
            si, ei = torchaudio.info(str(self.path))
            return si.length/si.rate
        
    @property
    def data(self): return self.spectro if self.spectro is not None else self.sig
    @data.setter
    def data(self, x):
        if self.spectro is not None: self.spectro = x
        else:                        self.sig = x
    
    @property
    def nsamples(self): 
        return self.sig.shape[-1]
    
    @property
    def nchannels(self): 
        return self.sig.shape[-2]
