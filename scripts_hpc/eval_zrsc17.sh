#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)
# The default partition is the 'general' partition
#SBATCH --partition=general
# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=short
# The default run (wall-clock) time is 1 minute
#SBATCH --time=01:30:00
# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=4
# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=18G
# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --gres=gpu
#SBATCH --mail-type=END

#subset="_first500" #"_first4k"
#subset_val="_first500"
#debug_flag="_debug"
#debug_flag=""
#if [ ! -z "$debug_flag" ]; then
#  debug_symbol="--debug"
#  dir_sym=""
#else
#  dir_sym="_all"
#  debug_symbol=""
#fi
#train_size="100"
train_size="360"
max_size_loaded="50000000"
eval_epoch=191

#~~~~Select one of the following lines~~~~~~#
PATH_CHECKPOINT=egs/librispeech/exp/train_clean_${train_size}_all_all_${max_size_loaded}_2GPU/checkpoint_${eval_epoch}.pt
##BELOW: TUNE LSTM Layers###
lr_suffix="_lr0.00005"
#PATH_CHECKPOINT=egs/librispeech/exp/tune_LSTMlayers/train_clean_${train_size}_lstm2_${max_size_loaded}_2GPU/checkpoint_${eval_epoch}.pt
#~~~~Select one of the above lines~~~~~~#

if [ "$eval_epoch" = "199" ]; then
  suffix=""
else
  suffix="ep_${eval_epoch}"
fi
source activate cpc_librilight
#dur_appoint=1s
#dur_appoint=10s
dur_appoint=120s

for zrlang in english ; do
  for dur in $dur_appoint ; do
    PATH_ITEM_FILE=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/relocated_from_DSP/zerospeech2017/data/test/${zrlang}/${dur}/${dur}.item #
    DATASET_PATH=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/relocated_from_DSP/zerospeech2017/data/test/${zrlang}/${dur}/
    PATH_OUT="$(dirname $PATH_CHECKPOINT)"
    mkdir -p $PATH_OUT/zrsc17${suffix}/${zrlang}_${dur}/
    if [ ! -f $PATH_OUT/zrsc17${suffix}/${zrlang}_${dur}/ABX_scores.json ]; then
    echo "Current progress: $zrlang, $dur:"
      python cpc/eval/ABX.py from_checkpoint $PATH_CHECKPOINT $PATH_ITEM_FILE $DATASET_PATH --seq_norm --strict --file_extension .wav --out $PATH_OUT/zrsc17${suffix}/${zrlang}_${dur}/ --cuda
    fi 
    # --get_encoded if activated using encoder output rather than AR output, meaning that by default AR output used for ABX.
    # --cuda, store_true, use GPU
    # --debug, store_true
  done
done
source deactivate

