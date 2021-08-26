#!/bin/sh
#you can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)
# The default partition is the 'general' partition
#SBATCH --partition=general
# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=short
# The default run (wall-clock) time is 1 minute
#SBATCH --time=02:00:00
# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1
# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=2
# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=2G
# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
##SBATCH --gres=gpu
#SBATCH --mail-type=END
##SBATCH --nodelist=cor1
#SBATCH --dependency=afterok:6616557

eval_set=dev-clean
eval_set=dev-other
eval_set=test-clean
eval_set=test-other
training_set=unlab_600
ep=199
#training_subset=_subset900utt
#training_subset=_subset3600utt
#training_subset=_subset7200utt
#training_subset=_subset14400utt
training_subset=""

###Deprecated
#model=egs/libri-light/exp/train_${training_set}${training_subset}_50000000_2GPU/checkpoint_${ep}.pt
#srun python extract_cpc_feat_to_kaldi.py $model /tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/data_eval/LibriSpeech/$eval_set $model/extract_feats/$eval_set --file_extension ".flac" --seq_norm
###Deprecated

#stage 0,1: feats.ark for eval and train respectively; GPU required
#stage 2,3: compose kaldi data folders eval and train respectively; GPU NOT required
#srun bash extract_cpc_feat_to_kaldi.sh --cpc-epoch $ep --train-set $training_set --subset-name "$training_subset" --stage 0 --stop-stage 1 --eval-set $eval_set 
#srun bash extract_cpc_feat_to_kaldi.sh --cpc-epoch $ep --train-set $training_set --subset-name "$training_subset" --stage 1 --stop-stage 2

#srun bash extract_cpc_feat_to_kaldi.sh --cpc-epoch $ep --train-set $training_set --subset-name "$training_subset" --stage 2 --stop-stage 3 --eval-set $eval_set 
#srun bash extract_cpc_feat_to_kaldi.sh --cpc-epoch $ep --train-set $training_set --subset-name "$training_subset" --stage 3 --stop-stage 4


##############Below considers extracting CPC representations of ZR17 test data sets.
# eval_language=english
eval_duration=120s
eval_duration=10s
eval_duration=1s
# subset_name="_subset900utt"
#srun bash extract_cpc_feat_to_kaldi_zr17.sh --stage 0 --stop-stage 1 --eval-duration $eval_duration --subset-name "_subset900utt"
#srun bash extract_cpc_feat_to_kaldi_zr17.sh --stage 0 --stop-stage 1 --eval-duration $eval_duration --subset-name "_subset3600utt"
#srun bash extract_cpc_feat_to_kaldi_zr17.sh --stage 0 --stop-stage 1 --eval-duration $eval_duration --subset-name "_subset7200utt"
#srun bash extract_cpc_feat_to_kaldi_zr17.sh --stage 0 --stop-stage 1 --eval-duration $eval_duration --subset-name "_subset14400utt"
#srun bash extract_cpc_feat_to_kaldi_zr17.sh --stage 0 --stop-stage 1 --eval-duration $eval_duration --subset-name ""

#srun bash extract_cpc_feat_to_kaldi_zr17.sh --stage 2 --stop-stage 3 --eval-duration $eval_duration --subset-name "_subset900utt"
#srun bash extract_cpc_feat_to_kaldi_zr17.sh --stage 2 --stop-stage 3 --eval-duration $eval_duration --subset-name "_subset3600utt"
#srun bash extract_cpc_feat_to_kaldi_zr17.sh --stage 2 --stop-stage 3 --eval-duration $eval_duration --subset-name "_subset7200utt"
#srun bash extract_cpc_feat_to_kaldi_zr17.sh --stage 2 --stop-stage 3 --eval-duration $eval_duration --subset-name "_subset14400utt"
srun bash extract_cpc_feat_to_kaldi_zr17.sh --stage 2 --stop-stage 3 --eval-duration $eval_duration --subset-name ""
