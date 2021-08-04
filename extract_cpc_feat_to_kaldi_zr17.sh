#!/bin/bash
# This script creates CPC representation of ZeroSpeech 2017 data and store them in the Kaldi format.
# The CPC model used here is trained with Libri-light dataset, either a subset or a full set. Refer to "subset_name".

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
source_audio=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/relocated_from_DSP/zerospeech2017/data
source_audio_ext=".wav"
options="--seq_norm --retain_extension" # by reference to /tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/relocated_from_DSP/zerospeech2017/kaldi_stuff/data/test/english/120s/wav.scp
#eval_set="dev-clean"
eval_language=english # english, french, mandarin
eval_duration=120s
. ./utils/parse_options.sh
. ./path.sh

checkpoint_path=$checkpoint_root/train_${train_set}${subset_name}${nlayers}_${max_size_loaded}${lr_suffix}${gpu_suffix}
checkpoint_file=checkpoint_${cpc_epoch}.pt
target_dir=$checkpoint_path/cpc_feats
original_kaldi_path=/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/kaldi/egs/relocated_from_DSP/zerospeech2017/kaldi_stuff #/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/libri-light/kaldi_related/
actual_train_set=${train_set}${subset_name}
if [ ! -f $checkpoint_path/$checkpoint_file ]; then
  echo "CPC Model not found: $checkpoint_path/$checkpoint_file"
  exit 0;
fi

if [  $stage -le 0 ] && [ $stop_stage -gt 0 ]; then
  echo "Convert features for ZR17 Test data  to .ark format"
  echo "CPC Model: $checkpoint_path/$checkpoint_file"
  for duration in $eval_duration ; do

    echo "Set: $eval_language/$duration;"
    echo "Output: $target_dir/zr17/test/$eval_language/$duration"
    echo "Input: $source_audio/test/$eval_language/$duration"
    echo "Extraction options: $options"
    if [ -f $target_dir/zr17/test/$eval_language/$duration/feats.ark ]; then
      echo "feats.ark exists in $target_dir/zr17/test/$eval_language/$duration/, refuses to overwrite..."
    else
      output_dir=$target_dir/zr17/test/$eval_language/$duration
      input_dir=$source_audio/test/$eval_language/$duration
      source activate cpc_librilight
      python extract_cpc_feat_to_kaldi.py --retain_extension $retain_extension $checkpoint_path/$checkpoint_file $input_dir $output_dir --file_extension $source_audio_ext $options
      source deactivate
    fi
  done
  
fi


if [  $stage -le 2 ] && [ $stop_stage -gt 2 ]; then
  for duration in $eval_duration ; do
    echo "Create feats.scp for $eval_language/$duration"
    utils/copy_data_dir.sh $original_kaldi_path/data/test/$eval_language/$duration  $target_dir/zr17/test/$eval_language/$duration/data_kaldi/ || exit 1
    rm -f $target_dir/zr17/test/$eval_language/$duration/data_kaldi/{feats.scp,cmvn.scp,utt2dur,utt2num_frames}
    copy-feats ark:$target_dir/zr17/test/$eval_language/$duration/feats.ark ark,scp:$target_dir/zr17/test/$eval_language/$duration/data_kaldi/feats.ark,$target_dir/zr17/test/$eval_language/$duration/data_kaldi/feats.scp
    feat-to-len scp:$target_dir/zr17/test/$eval_language/$duration/data_kaldi/feats.scp ark,t:$target_dir/zr17/test/$eval_language/$duration/data_kaldi/utt2num_frames
    steps/compute_cmvn_stats.sh $target_dir/zr17/test/$eval_language/$duration/data_kaldi
    utils/validate_data_dir.sh  --no-text $target_dir/zr17/test/$eval_language/$duration/data_kaldi || exit 1

  done
  
fi


echo "Succeeded"
