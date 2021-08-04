#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)
# The default partition is the 'general' partition
#SBATCH --partition=general
# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=short
# The default run (wall-clock) time is 1 minute
#SBATCH --time=04:00:00
# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=8
# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=20G
# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --gres=gpu
#SBATCH --mail-type=END

#subset="_first500" #"_first4k"
subset_val="_first500"
#debug_flag="_debug"
debug_flag=""
if [ ! -z "$debug_flag" ]; then
  debug_symbol="--debug"
  dir_sym=""
else
  dir_sym="_all"
  debug_symbol=""
fi
train_size="100"
max_size_loaded="100000000"
PATH_AUDIO_FILES=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/english/LibriSpeech/train-clean-${train_size}/
#PATH_AUDIO_FILES=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/english/LibriSpeech/train-clean-${train_size}/
PATH_CHECKPOINT_DIR=egs/librispeech/exp/train_clean_${train_size}${debug_flag}${dir_sym}${dir_sym}/
mkdir -p $PATH_CHECKPOINT_DIR || exit 1
TRAINING_SET=egs/librispeech/train_clean_${train_size}_tr_cv/train_split${subset}.txt
VAL_SET=egs/librispeech/train_clean_${train_size}_tr_cv/valid_split${subset_val}.txt
EXTENSION=".flac"
source activate cpc_librilight
python cpc/train.py --pathDB $PATH_AUDIO_FILES --pathCheckpoint $PATH_CHECKPOINT_DIR --pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION --nGPU 1 --n_process_loader 8 ${debug_symbol} --restart --max_size_loaded $max_size_loaded #--debug
source deactivate
#python cpc/train.py --pathCheckpoint $PATH_CHECKPOINT_DIR --nGPU 22 --n_process_loader 10
