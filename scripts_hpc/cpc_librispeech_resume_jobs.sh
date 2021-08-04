#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)
# The default partition is the 'general' partition
#SBATCH --partition=general
# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=long
# The default run (wall-clock) time is 1 minute
#SBATCH --time=96:00:00
# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=4
# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=8G
# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --gres=gpu:2
#SBATCH --mail-type=END
##SBATCH --nodelist=cor1
#SBATCH --exclude=wis1
subset="_first4k" #"_first4k"
#subset_val="_first500"
#debug_flag="_debug"
#debug_flag=""

#train_size="100"
train_size="360"

max_size_loaded="50000000"
PATH_AUDIO_FILES=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/english/LibriSpeech/train-clean-${train_size}/
#PATH_AUDIO_FILES=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/corpora/english/LibriSpeech/train-clean-${train_size}/
nGPU=2
#PATH_CHECKPOINT_DIR=egs/librispeech/exp/train_clean_${train_size}${debug_flag}_${max_size_loaded}${subset}_6GPU/
PATH_CHECKPOINT_DIR=egs/librispeech/exp/train_clean_${train_size}_all_all_50000000_2GPU
#PATH_CHECKPOINT_DIR=egs/librispeech/exp/train_clean_${train_size}_all_all_50000000_4GPU
stop_epoch=200
source activate cpc_librilight
python cpc/train.py  --pathCheckpoint $PATH_CHECKPOINT_DIR  --nGPU ${nGPU} --n_process_loader 8  --batchSizeGPU 16 --stop_epoch $stop_epoch #--pathTrain $TRAINING_SET --pathVal $VAL_SET --file_extension $EXTENSION --nGPU ${nGPU} --n_process_loader 8 ${debug_symbol} --restart --max_size_loaded $max_size_loaded --batchSizeGPU 16
source deactivate
