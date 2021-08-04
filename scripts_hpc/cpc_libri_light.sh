#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)
# The default partition is the 'general' partition
#SBATCH --partition=general
# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=long
# The default run (wall-clock) time is 1 minute
#SBATCH --time=91:00:00
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
#SBATCH --nodelist=cor1

train_set=600

#subset="_subset900utt" #""
#subset="_subset3600utt" #""
#subset="_subset7200utt" #""
subset="_subset14400utt" #""
#subset="" # -> unlab-600 full set


#subset_val="_first500"
#debug_flag="_debug"
debug_flag=""
if [ ! -z "$debug_flag" ]; then
  debug_symbol="--debug"
  dir_sym=""
else
  dir_sym="_all"
  debug_symbol=""
fi
max_size_loaded="50000000"
PATH_AUDIO_FILES=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/data_unlab/unlab-${train_set}_cut/ #/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/english/LibriSpeech/train-clean-${train_size}/
#PATH_AUDIO_FILES=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/english/LibriSpeech/train-clean-${train_size}/
nGPU=2
PATH_CHECKPOINT_DIR=egs/libri-light/exp/train_unlab_${train_set}${subset}${debug_flag}_${max_size_loaded}_${nGPU}GPU/
mkdir -p $PATH_CHECKPOINT_DIR || exit 1
TRAINING_SET=egs/libri-light/train_unlab_${train_set}${subset}/train_split.txt #librispeech/train_clean_${train_size}_tr_cv/train_split${subset}.txt
VAL_SET=egs/libri-light/train_unlab_${train_set}${subset}/valid_split.txt #egs/librispeech/train_clean_${train_size}_tr_cv/valid_split${subset_val}.txt
EXTENSION=".flac"

stop_epoch=200 #100
save_step=1

source activate cpc_librilight

# start a new job
#python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION --nGPU ${nGPU} --n_process_loader 8 ${debug_symbol} --restart --max_size_loaded $max_size_loaded --batchSizeGPU 16 --stop_epoch $stop_epoch --save_step $save_step

# resume a job:
python cpc/train.py  --pathCheckpoint $PATH_CHECKPOINT_DIR  --nGPU ${nGPU} --n_process_loader 8  --batchSizeGPU 16 --stop_epoch $stop_epoch --save_step $save_step

source deactivate


#python cpc/train.py --pathCheckpoint $PATH_CHECKPOINT_DIR --nGPU 22 --n_process_loader 10
