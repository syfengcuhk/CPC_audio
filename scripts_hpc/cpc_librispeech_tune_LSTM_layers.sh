#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)
# The default partition is the 'general' partition
#SBATCH --partition=general
# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=long
# The default run (wall-clock) time is 1 minute
#SBATCH --time=10:30:00
# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=6
# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=6G
# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --gres=gpu:2
#SBATCH --mail-type=END
##SBATCH --nodelist=awi02


train_size="100"
#train_size="360"

max_size_loaded="50000000"
PATH_AUDIO_FILES=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/english/LibriSpeech/train-clean-${train_size}/
nGPU=2
lstm_layers=2
learningRate="0.00005"
PATH_CHECKPOINT_DIR=egs/librispeech/exp/tune_LSTMlayers/train_clean_${train_size}_lstm${lstm_layers}_${max_size_loaded}_${nGPU}GPU/
mkdir -p $PATH_CHECKPOINT_DIR || exit 1
TRAINING_SET=egs/librispeech/train_clean_${train_size}_tr_cv/train_split.txt
VAL_SET=egs/librispeech/train_clean_${train_size}_tr_cv/valid_split.txt
EXTENSION=".flac"
stop_epoch=200
save_step=1
source activate cpc_librilight
## start a new job:
#python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION --nGPU ${nGPU} --n_process_loader 8 ${debug_symbol} --restart --max_size_loaded $max_size_loaded --batchSizeGPU 16 --stop_epoch ${stop_epoch} --save_step $save_step --nLevelsGRU $lstm_layers --learningRate ${learningRate}

## resume a job:
python cpc/train.py  --pathCheckpoint $PATH_CHECKPOINT_DIR  --nGPU ${nGPU} --n_process_loader 8  --batchSizeGPU 16 --stop_epoch $stop_epoch --save_step $save_step  --nLevelsGRU $lstm_layers --learningRate ${learningRate}
source deactivate
