#PRJPATH="$(dirname $( cd "$(dirname "$0")" ; pwd -P ))"
#echo $PRJPATH
#PYTHONPATH=$PRJPATH/grpc-services/gen-py python $PRJPATH/server/server.py $@

#python server.py --model_config conf/decode_engine.yaml --host 0.0.0.0 --port 1234 --vad_aggressiveness 3
#python server.py --model_config conf/decode_engine_V2.yaml --host 0.0.0.0 --port 1234 --vad_aggressiveness 3
python server.py --model_config conf/decode_engine_V3.yaml --host 0.0.0.0 --port 9876 --vad_aggressiveness 3
