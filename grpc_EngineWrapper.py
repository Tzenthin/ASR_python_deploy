# Wrapper class that adds pre/post-processing to STT engine

import logging
from attrdict import AttrDict
import os
from datetime import datetime
import wave
from text2digits import text2digits
import configargparse

class EngineWrapper(object):
    def __init__(self, config, engine):
        parser = configargparse.ArgumentParser(description="STT GRPC engine wrapper.",
                default_config_files=["config"])
        parser.add_argument('--savewav', default='',
                help="Save .wav files of utterences to given directory.")
        ARGS, _ = parser.parse_known_args()
        args = vars(ARGS)
        self.config = AttrDict({**args, **config})
        if self.config.savewav: os.makedirs(self.config.savewav, exist_ok=True)
        self.engine = engine
        self.logger = logging.getLogger('wrapper.save_post')

    def post_fun(self, result):
        if isinstance(result, dict):
            text = result.get('transcript', '')
            result['transcript'] = text2digits.Text2Digits().convert(text)
            return(result)
        else:
            return(text2digits.Text2Digits().convert(result))

    def decode_audio(self, audio):
        if self.config.savewav:
            self.save_wave(audio)
        text = self.engine.decode_audio(audio)
        return self.post_fun(text)

    def get_stream(self, result_queue):
        # FIXME: Should include some guard agains very long audio!
        if self.config.savewav:
            self.audio = bytearray()
        # FIXME: Poor workaround for post_fun?
        pre_wrapper = result_queue.put
        def put_wrapper(item, *args, **kwargs):
            pre_wrapper(self.post_fun(item), *args, **kwargs)
        result_queue.put = put_wrapper
        return self.engine.get_stream(result_queue)

    def feed_audio_data(self, stream, audio):
        # FIXME: Will different stream have an issue?
        if self.config.savewav:
            self.audio.extend(audio)
        return self.engine.feed_audio_data(stream, audio)

    def finish_stream(self, stream):
        if self.config.savewav:
            self.save_wave(self.audio)
        return self.engine.finish_stream(stream)

    def __getattr__(self, attr):
        return getattr(self.engine, attr)
