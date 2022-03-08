class STTEngine(object):
    def __init__(self, model_folder):
        raise NotImplementedError

    def decode_audio(self, audio):
        raise NotImplementedError

    def get_stream(self, result_queue):
        raise NotImplementedError

    def feed_audio_data(self, stream, audio):
        raise NotImplementedError

    def finish_stream(self, stream):
        raise NotImplementedError

    def check_compatibility(self, config):
        raise NotImplementedError
