import logging
import logging.config
from concurrent import futures
import time

import grpc
#import grpc_services.gen_py.stt_pb2_grpc as service
#import grpc_services.gen_py.stt_pb2 as messages
import stt_pb2_grpc as service
import stt_pb2 as messages
from grpc_WenetEngine import WenetEngine as Engine
from grpc_EngineWrapper import EngineWrapper
from grpc_VADWrapper import VADWrapper

from queue import Queue

_ONE_DAY_IN_SECONDS = 60 * 60 * 24 

def service_decorator(fun):
    """ Wraps services to raise grpc.StatusCode on exception.
        Logs the actual exception for debugging.
    """
    def wrapped(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except MemoryError:
            args[0].logger.exception('Exception occurred.')
            context = args[2]
            context.set_details("Number of simultaneous requests exceeded.")
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
        except Exception as e:
            args[0].logger.exception('Exception occurred.')
            context = args[2]
            context.set_details("Unknown error occured.")
            context.set_code(grpc.StatusCode.ABORTED)
    return wrapped

def service_decorator_gen(fun):
    """ Wraps services to raise grpc.StatusCode on exception.
        Logs the actual exception for debugging.
        FIXME: To be combined with service_decorator!
    """
    def wrapped(*args, **kwargs):
        try:
            gen = fun(*args, **kwargs)
            while True:
                try:
                    g_next = next(gen)
                except StopIteration:
                    break
                else:
                    yield g_next
        except MemoryError:
            args[0].logger.exception('Exception occurred.')
            context = args[2]
            context.set_details("Number of simultaneous requests exceeded.")
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
        except Exception as e:
            args[0].logger.exception('Exception occurred.')
            context = args[2]
            context.set_details("Unknown error occured.")
            context.set_code(grpc.StatusCode.ABORTED)
    return wrapped

class STTService(service.STTServicer):
    def __init__(self, engine_config_path, vad_aggressiveness=None):
        super(STTService, self).__init__()
        self.logger = logging.getLogger('server')
        self.config = {
                'vad_aggressiveness': vad_aggressiveness
                }
        self.__models = [Engine(engine_config_path)]

    @service_decorator
    def Recognize(self, request, context):
        engine = self.configure_engine(request.config, context)
        if engine is None:
            return messages.RecognizeResponse()
        text = engine.decode_audio(request.audio.content)
        result = messages.RecognizeResponse(transcript=text)
        return result

    @service_decorator_gen
    def StreamingRecognize(self, request_iterator, context):
        configured = False
        for message in request_iterator:
            if message.WhichOneof("streaming_request") == "config":
                print("streaming_config", message.WhichOneof("streaming_request"))
                engine = self.configure_engine(message.config, context)
                if engine is None:
                    return
                queue = Queue()
                try:
                    stream = engine.get_stream(queue)
                except NotImplementedError:
                    context.set_details("Number of simultaneous requests exceeded.")
                    context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                configured = True
 
            elif message.WhichOneof("streaming_request") == "audio_content":
                t1=time.time()
                if not configured:
                    context.set_details("'streaming_config' not recieved before 'streaming_request'.")
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    break
                engine.feed_audio_data(stream, message.audio_content)
                while queue.qsize()>0: 
                    text = queue.get()
                    response = messages.StreamingRecognizeResponse(transcript=text)
                    yield response
        if configured:
            engine.finish_stream(stream)
            text = ' '.join(list(queue.queue))
        else:
            text = ''
        result = messages.StreamingRecognizeResponse(transcript=text)
        yield result


    def configure_engine(self, recognition_config, context):
        """Returns requested engine or None with grpc.StatusCode updated in context"""
        # FIXME: Add load balancer here.
        # Adding two dictionary, preventing overwriting of base config
        config = {}
        for field, value in recognition_config.ListFields():
            config[field.name] = value
        config = {**(config), **self.config}
        for i, model in enumerate(self.__models):
            if model.check_compatibility(config):
                model = VADWrapper(config=config, engine=model) if config.get('vad_aggressiveness', None) else model
                self.logger.info('Configured engine {} for {}Hz.'.format(i, recognition_config.sample_rate_hertz))
                return EngineWrapper(config=config, engine=model)
        context.set_details("Invalid RecognitionConfig.")
        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        self.logger.error('Invalid config provided: {}'.format(config))
        return None

def serve(port, model_config_path, vad_aggressiveness):
    # FIXME: number of workers limit the max number of streams!
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix='gRPCThread'))
    service.add_STTServicer_to_server(STTService(model_config_path, vad_aggressiveness), server)
    server.add_insecure_port(port)
    server.start()
    print('Server started on ' + port)
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        print('Server on ' + port + ' stopped')

if __name__ == '__main__':
    import configargparse
    parser = configargparse.ArgumentParser(description="STT GRPC server.",
            default_config_files=["config"])
    parser.add_argument('--model_config',required=True, help='config file path')
    # Server parameters
    parser.add_argument('--host', default='0.0.0.0',
            help='Host IP address for running STT engine.')
    parser.add_argument('--port', type=int, default=50051,
            help='Host port running STT engine.')
    parser.add_argument('--logconf', default='',
            help="Logging.conf file with server, engine, wrapper loggers")
    parser.add_argument('-v', '--vad_aggressiveness', type=int, default=None,
            help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: None")

    ARGS, _ = parser.parse_known_args()
    if ARGS.logconf:
        logging.config.fileConfig(ARGS.logconf)
    serve('{}:{}'.format(ARGS.host, ARGS.port), ARGS.model_config, ARGS.vad_aggressiveness)
