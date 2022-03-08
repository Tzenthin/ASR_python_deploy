import logging
import argparse
import os
import numpy as np
import tempfile
import csv
import math
import yaml
import time
import copy
from grpc_STTEngine import STTEngine
from collections import deque
from wenet.utils.file_utils import read_symbol_table
from wenet.transformer.asr_model_streaming import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import datetime
import wave

class WenetEngine(STTEngine):
    DECODE_CHUNK_SIZE = 4000

    def __init__(self, model_config_path):
        """ Loads and sets up the model
        """
        self.logger = logging.getLogger('engine.wenet!')
        with open(model_config_path, 'r') as fin:
            self.configs = yaml.load(fin, Loader=yaml.FullLoader)
        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s')
        symbol_table = read_symbol_table(self.configs['dict_path'])
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.configs['gpu'])
        decode_conf = copy.deepcopy(self.configs['data_conf'])
        decode_conf['filter_conf']['max_length'] = 102400
        decode_conf['filter_conf']['min_length'] = 0
        decode_conf['filter_conf']['token_max_length'] = 102400
        decode_conf['filter_conf']['token_min_length'] = 0
        use_cuda = self.configs['gpu'] >= 0 and torch.cuda.is_available()
        #self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.device = torch.device('cpu')
        # convert num to symbles 
        self.num2sym_dict = {}
        with open(self.configs['dict_path'], 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                self.num2sym_dict[int(arr[1])] = arr[0]
        self.eos = len(self.num2sym_dict) - 1

        self.models = deque(maxlen=self.configs['engine_max_decoders'])
        asr = init_asr_model(self.configs)
        load_checkpoint(asr, self.configs['model_path'])
        asr = asr.to(self.device)
        asr.eval()
        for i in range(self.configs['engine_max_decoders']):
            self.models.append(asr)
            self.logger.info('Model {} loaded.'.format(id(asr)))
        self.streams = []

    def _get_model(self):
        """ Retrieves a free asr.
        """
        if len(self.models):
            model = self.models.pop()
            self.logger.info('Model {} engaged.'.format(id(model)))
            return model
        else:
            for ix, s in enumerate(self.streams):
                if (time.time() - s['last_activity']) > self.configs['engine_max_inactivity_secs']:
                    model = s['model']
                    self.streams.pop(ix)
                    self.logger.info('Model {} force freed.'.format(id(model)))
                    return model
        raise MemoryError

    def _free_model(self, model):
        self.models.append(model)
        self.logger.info('Model {} freed.'.format(id(model)))

    def _num2sym(self, hyps):
        content = ''
        for w in hyps:
            if w == self.eos:
                break
            content += self.num2sym_dict[w]
        return content

    def _feature_extraction(self, waveform): 
        num_mel_bins = self.configs['data_conf']['fbank_conf']['num_mel_bins'] # 80
        frame_length = self.configs['data_conf']['fbank_conf']['frame_length'] # 25
        frame_shift = self.configs['data_conf']['fbank_conf']['frame_shift']   # 10
        dither = self.configs['data_conf']['fbank_conf']['dither'] # 0.0
        feat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither, 
                          energy_floor=0.0,
                          sample_frequency=self.configs['engine_sample_rate_hertz'])
        feat = feat.unsqueeze(0) #.to(device)
        feat_length = torch.IntTensor([feat.size()[1]])
        return feat, feat_length

    def decode_audio(self, audio): #一句话解码
        if len(audio)<1600: #小于0.1秒，不解码
            return ''
        waveform = np.frombuffer(audio, dtype=np.int16)
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        waveform = waveform.to(self.device)
        waveform_feat, feat_length = self._feature_extraction(waveform)
        model = self._get_model()
        with torch.no_grad():
            hyps, scores = model.recognize(waveform_feat,
                                 feat_length,
                                 beam_size=self.configs['beam_size'],
                                 decoding_chunk_size=-1,
                                 num_decoding_left_chunks=self.configs['num_decoding_left_chunks'],
                                 simulate_streaming=True) 
            hyps = [hyp.tolist() for hyp in hyps[0]]
            result = self._num2sym(hyps)
        print(result)
        self._free_model(model)
        if len(audio)>WenetEngine.DECODE_CHUNK_SIZE:
            self.save_wave(audio, result)
        return result


    def get_stream(self, result_queue):
        """ Establishes stream to model.
        """
        asr = self._get_model()
        stream = {'model':asr,
                'current_audio':bytes(), 
                'chunk_size':0, 
                'total_audio_len':0,
                'last_activity':time.time(), 
                'intermediate':'', 
                'result_queue':result_queue}
        self.streams.append(stream)
        self.logger.info('Stream established to Model {}.'.format(id(asr)))
        return stream


    def feed_audio_streaming(self, stream, audio):
        stream['last_activity'] = time.time()
        stream['current_audio'] += audio
        stream['chunk_size'] += len(audio)

    def decode_audio_streaming(self, stream):
        if stream['chunk_size'] >= WenetEngine.DECODE_CHUNK_SIZE:
            waveform = np.frombuffer(stream['current_audio'], dtype=np.int16)
            waveform = torch.from_numpy(waveform).float().unsqueeze(0)
            waveform = waveform.to(self.device)
            waveform_feat, feat_length = self._feature_extraction(waveform)
            with torch.no_grad():
                hyps, scores = stream['model'].recognize(waveform_feat,
                                 feat_length,
                                 beam_size=self.configs['beam_size'],
                                 decoding_chunk_size=-1,
                                 num_decoding_left_chunks=self.configs['num_decoding_left_chunks'],
                                 simulate_streaming=True) 
            hyps = [hyp.tolist() for hyp in hyps[0]]
            result = self._num2sym(hyps)
            print(result)
            self.save_wave(stream['current_audio'], result)
            stream['result_queue'].put(result)
            stream['chunk_size'] = 0
            stream['current_audio'] = b''


    def finish_stream(self, stream):
        """ Finishes decoding destroying stream.
        """
        asr = stream['model']
        self.logger.info('Audio of length {} processed in stream to Model {}.'.format(stream['total_audio_len'], id(asr)))
        self._free_model(stream['model'])

    def check_compatibility(self, config):
        """ Checks if engine is compatible with given config.
        Args:
            config: Key, value pairs of requested features.
        Returns:
            boolean, True if engine matches config.
        """
        if 'sample_rate_hertz' in config:
            return config['sample_rate_hertz'] == self.configs['engine_sample_rate_hertz']
        return True


    def save_wave(self, audio, transcript):
        now_time = datetime.datetime.now() 
        year = str(now_time.year) 
        month = str(now_time.month) 
        day = str(now_time.day) 
        audio_dir = os.path.join(self.configs['audio_save_path'], year, month, day)
        wav_file = audio_dir+'/'+ datetime.datetime.now().strftime("%H-%M-%S-%f_") + transcript[0:20] + '_.wav'
        if not os.path.exists(audio_dir):
            os.makedirs(audio_dir)
        with wave.open(wav_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.configs['engine_sample_rate_hertz'])
            wf.writeframes(audio)

