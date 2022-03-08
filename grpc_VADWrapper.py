# Wrapper class that adds vad pre-processing to STT engine

import logging
from attrdict import AttrDict
import webrtcvad
import collections

class VADAudio():
    SAMPLE_WIDTH = 2 # Number of bytes for each sample
    CHANNELS = 1

    #def __init__(self, aggressiveness, rate, frame_duration_ms, padding_ms=300, padding_ratio=0.75):
    def __init__(self, aggressiveness, rate, frame_duration_ms, padding_ms=200, padding_ratio=0.4):
        """Initializes VAD with given aggressivenes and sets up internal queues"""
        self.vad = webrtcvad.Vad(aggressiveness)
        self.rate = rate
        self.frame_duration_ms = frame_duration_ms
        self._frame_length = int( rate * (frame_duration_ms/1000.0) * self.SAMPLE_WIDTH )
        self._buffer_queue = collections.deque()
        self.ring_buffer = collections.deque(maxlen = padding_ms // frame_duration_ms)
        self._ratio = padding_ratio
        self.triggered = False

    def add_audio(self, audio):
        """Adds new audio to internal queue"""
        for x in audio:
            self._buffer_queue.append(x)

    def frame_generator(self):
        """Generator that yields audio frames of frame_duration_ms"""
        while len(self._buffer_queue) > self._frame_length:
            frame = bytearray()
            for _ in range(self._frame_length):
                frame.append(self._buffer_queue.popleft())
            yield bytes(frame)

    def vad_collector(self):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        for frame in self.frame_generator():
            is_speech = self.vad.is_speech(frame, self.rate)
            if not self.triggered:
                self.ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in self.ring_buffer if speech])
                if num_voiced > self._ratio * self.ring_buffer.maxlen:
                    self.triggered = True
                    for f, s in self.ring_buffer:
                        yield f
                    self.ring_buffer.clear()
            else:
                yield frame
                self.ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                if num_unvoiced > self._ratio * self.ring_buffer.maxlen:
                    self.triggered = False
                    yield None
                    self.ring_buffer.clear()

class VADWrapper(object):
    def __init__(self, config, engine):
        """ Initializes the object.

        Args:
            config (dict): Key, value pair of configuration values.

        Returns:
            STTEngine object with pre-processing decorators.
        """
        if 'vad_aggressiveness' not in config:
            raise ValueError('vad_aggressiveness not provided')
        self.vad = VADAudio(config['vad_aggressiveness'], config['sample_rate_hertz'], 20)
        self.engine = engine
        self.logger = logging.getLogger('wrapper.vad')

    def decode_audio(self, audio):
        # FIXME: Assert single inference via EngineWrapper.
        self.vad.add_audio(audio)
        audio = b''.join(f for f in self.vad.vad_collector() if f is not None)
        #print(len(audio))
        return self.engine.decode_audio(audio)

    def get_stream(self, result_queue):
        self._stream = self.engine.get_stream(result_queue)
        return {'stream': self._stream, 'result_queue':result_queue}

    def feed_audio_data(self, stream, audio):
        # FIXME: Assert single inference via EngineWrapper.
        self.vad.add_audio(audio)
        temp_audio = b''
        #print('%'*23, stream['stream']['current_audio'])
        for frame in self.vad.vad_collector():
            if frame is None:
                # VAD detected end of speech. Finish and start a new stream
                if len(temp_audio)> 0: #16000*0.1: #大于0.2秒才进行解码
                    self.engine.feed_audio_streaming(stream['stream'], temp_audio)
                    self.engine.decode_audio_streaming(stream['stream'])
                temp_audio = b''
            else:
                temp_audio += frame
        if temp_audio != b'':
            self.engine.feed_audio_streaming(stream['stream'], temp_audio)


    def finish_stream(self, stream):
        """ Finishes decoding destroying stream.
        """
        return self.engine.finish_stream(stream['stream'])

    def __getattr__(self, attr):
        """ Passess all non-implemented method to engine
        """
        return getattr(self.engine, attr)
