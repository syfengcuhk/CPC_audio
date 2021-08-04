#!/bin/bash
# This script creates CPC representation of Libri-light training and evaluation data and store them in the Kaldi format.

cpc_epoch=199
checkpoint_root=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/CPC_audio/egs/libri-light/exp/tune_LSTMlayers/
train_set=unlab_600
subset_name="_subset900utt"
max_size_loaded="50000000"
nlayers=2
lr_suffix="_lr0.00005"
gpu_suffix="_2GPU"
stage=0
stop_stage=1
source_audio=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/
source_audio_ext=".flac"
options="--seq_norm"
eval_set="dev-clean"
. ./utils/parse_options.sh
. ./path.sh

checkpoint_path=$checkpoint_root/train_${train_set}${subset_name}${nlayers}_${max_size_loaded}${lr_suffix}${gpu_suffix}
checkpoint_file=checkpoint_${cpc_epoch}.pt
target_dir=$checkpoint_path/cpc_feats
original_kaldi_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/
train_wav_scp_path=$original_kaldi_path/data/train_${train_set}${subset_name}
actual_train_set=${train_set}${subset_name}
if [ ! -f $checkpoint_path/$checkpoint_file ]; then
  echo "CPC Model not found: $checkpoint_path/$checkpoint_file"
  exit 0;
fi

if [  $stage -le 0 ] && [ $stop_stage -gt 0 ]; then
  echo "Convert features for $eval_set  to .ark format"
  echo "CPC Model: $checkpoint_path/$checkpoint_file"
  for x in $eval_set ; do
    echo "Set: $x;"
    echo "Output: $target_dir/libri_light/${x//-/_}"
    echo "Input: $source_audio/data_eval/LibriSpeech/$x"
    echo "Extraction options: $options"
    if [ -f $target_dir/libri_light/${x//-/_}/feats.ark ]; then
      echo "feats.ark exists in $target_dir/libri_light/${x//-/_}, refuses to overwrite..."
    else
      output_dir=$target_dir/libri_light/${x//-/_}
      input_dir=$source_audio/data_eval/LibriSpeech/$x
      source activate cpc_librilight
      python extract_cpc_feat_to_kaldi.py $checkpoint_path/$checkpoint_file $input_dir $output_dir --file_extension ".flac" $options
      source deactivate
    fi
  done
  
fi

if [  $stage -le 1 ] && [ $stop_stage -gt 1 ]; then
  echo "Convert features for training set: train_$actual_train_set to .ark format"
  echo "CPC Model: $checkpoint_path/$checkpoint_file"
  output_dir=$target_dir/libri_light/train_$actual_train_set
  echo "wav.scp: $train_wav_scp_path/wav.scp"
  if [ -f $target_dir/libri_light/train_$actual_train_set/feats.ark ]; then
    echo "feats.ark exists in $target_dir/libri_light/train_$actual_train_set, refuses to overwrite..."
  else
    source activate cpc_librilight 
    python extract_cpc_feat_to_kaldi.py $checkpoint_path/$checkpoint_file dummy $output_dir --wav_scp $train_wav_scp_path/wav.scp --file_extension ".flac" $options
    source deactivate 
  fi
fi

if [  $stage -le 2 ] && [ $stop_stage -gt 2 ]; then
  echo "Create feats.scp for $eval_set"
  for x in $eval_set; do
    utils/copy_data_dir.sh $original_kaldi_path/data/$x  $target_dir/libri_light/${x//-/_}/data_kaldi/ || exit 1
    rm -f $target_dir/libri_light/${x//-/_}/data_kaldi/{feats.scp,cmvn.scp,utt2dur,utt2num_frames}
    copy-feats ark:$target_dir/libri_light/${x//-/_}/feats.ark ark,scp:$target_dir/libri_light/${x//-/_}/data_kaldi/feats.ark,$target_dir/libri_light/${x//-/_}/data_kaldi/feats.scp
    feat-to-len scp:$target_dir/libri_light/${x//-/_}/data_kaldi/feats.scp ark,t:$target_dir/libri_light/${x//-/_}/data_kaldi/utt2num_frames
    steps/compute_cmvn_stats.sh $target_dir/libri_light/${x//-/_}/data_kaldi/
    utils/validate_data_dir.sh  --no-text $target_dir/libri_light/${x//-/_}/data_kaldi || exit 1

  done
  
fi

if [  $stage -le 3 ] && [ $stop_stage -gt 3 ]; then
  echo "Create feats.scp for training set: train_$actual_train_set"
  utils/copy_data_dir.sh $original_kaldi_path/data/train_$actual_train_set $target_dir/libri_light/train_$actual_train_set/data_kaldi || exit 1
  rm -f $target_dir/libri_light/train_$actual_train_set/data_kaldi/{feats.scp,cmvn.scp,utt2dur,utt2num_frames}
  copy-feats ark:$target_dir/libri_light/train_$actual_train_set/feats.ark ark,scp:$target_dir/libri_light/train_$actual_train_set/data_kaldi/feats.ark,$target_dir/libri_light/train_$actual_train_set/data_kaldi/feats.scp
  feat-to-len scp:$target_dir/libri_light/train_$actual_train_set/data_kaldi/feats.scp ark,t:$target_dir/libri_light/train_$actual_train_set/data_kaldi/utt2num_frames
  steps/compute_cmvn_stats.sh $target_dir/libri_light/train_$actual_train_set/data_kaldi/
  utils/validate_data_dir.sh  --no-text $target_dir/libri_light/train_$actual_train_set/data_kaldi
fi


echo "Succeeded"
