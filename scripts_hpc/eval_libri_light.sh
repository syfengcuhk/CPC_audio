#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)
# The default partition is the 'general' partition
#SBATCH --partition=general
# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=short
# The default run (wall-clock) time is 1 minute
#SBATCH --time=01:10:00
# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=4
# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=8G
# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --gres=gpu
#SBATCH --mail-type=END

#train_size="100"
train_size="360"
max_size_loaded="50000000"
eval_epoch=191

#~~~~~~~#
PATH_CHECKPOINT=egs/librispeech/exp/train_clean_${train_size}_all_all_50000000_2GPU/checkpoint_${eval_epoch}.pt #train_clean_${train_size}${debug_flag}${dir_sym}${dir_sym}/
#PATH_CHECKPOINT=egs/librispeech/exp/tune_LSTMlayers/train_clean_${train_size}_lstm2_${max_size_loaded}_2GPU/checkpoint_${eval_epoch}.pt
#~~~~~~~#

if [ "$eval_epoch" = "199" ]; then
  suffix=""
else
  suffix="ep_${eval_epoch}"
fi
source activate cpc_librilight

#eval_set_appoint=dev-clean
#eval_set_appoint=dev-other
#eval_set_appoint=test-clean
eval_set_appoint=test-other
for eval_set in $eval_set_appoint ; do
    PATH_ITEM_FILE=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/eval/ABX_src/ABX_data/${eval_set}.item
    DATASET_PATH=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/data_eval/LibriSpeech/$eval_set/
    PATH_OUT="$(dirname $PATH_CHECKPOINT)"
    mkdir -p $PATH_OUT/libri_light${suffix}/${eval_set}/
    if [ ! -f $PATH_OUT/libri_light${suffix}/${eval_set}/ABX_scores.json ]; then
    echo "Current progress: $eval_set:"
      python cpc/eval/ABX.py from_checkpoint $PATH_CHECKPOINT $PATH_ITEM_FILE $DATASET_PATH --seq_norm --strict --file_extension .flac --out $PATH_OUT/libri_light${suffix}/${eval_set}/ --cuda
    fi 
    # --get_encoded if activated using encoder output rather than AR output
    # --cuda, store_true, use GPU
    # --debug, store_true
done
source deactivate

