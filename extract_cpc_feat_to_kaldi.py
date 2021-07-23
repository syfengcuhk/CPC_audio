import argparse
import sys
import torch
import os
from cpc.feature_loader import buildFeature, FeatureModule, loadModel
sys.path.append("/tudelft.net/staff-bulk/ewi/insy/SpeechLab/siyuanfeng/software/CPC_audio")
import numpy as np

def parseArgs(argv):
    parser = argparse.ArgumentParser(description='Extract CPC features to Kaldi format')
    parser.add_argument('checkpoint_path', type=str, help='Path to the CPC model to generate CPC representation')
    parser.add_argument('path_dataset', type=str, help="Path to the dataset")
    parser.add_argument('output_dir', type=str, help='Path where the kaldi format results feats.ark is saved')
    parser.add_argument('--get_encoded', action='store_true', help='If activated, use CPC encoder output instead of AR output')
    parser.add_argument('--file_extension', type=str, default='.wav', help="Extension of audio file")
    parser.add_argument('--seq_norm', action='store_true', help="If activated, normalize each batch of feature across time channel.") 
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parseArgs(argv)
    # Load CPC model -> FeatureModule -> buildFeature -> write to text file feats.ark
    print("Checkpoint path: %s; Use encoder: %r" % ( args.checkpoint_path, args.get_encoded ) )
    print("Output directory: %s" % args.output_dir )
    print("Dataset: %s; extension: %s" % (args.path_dataset, args.file_extension ))
    model = loadModel([args.checkpoint_path])[0]
    feature_maker = FeatureModule(model, args.get_encoded).cuda().eval()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "feats.ark"), 'w') as f_write:

        for root, dirs, files in os.walk(args.path_dataset):
            for name in files:
                i = 1
                if name.endswith(args.file_extension):
                    #print(os.path.join(root,name))
                    audiofile = os.path.join(root,name)
                    uttname = name.split('.')[0]
#                    print(uttname)
                    feature_this_utterance = buildFeature(feature_maker, audiofile, seqNorm=args.seq_norm , strict=True) # a torch tensor of dimension [1, #frames, #dimsions]
                    feature_to_store = feature_this_utterance.view(-1,feature_this_utterance.size()[-1]).cpu()  # dimension to [#frames, #dimensions]
                    np_feature_to_store = feature_to_store.numpy()
                    f_write.write(uttname + " [\n")
                    for item_np in np_feature_to_store[:-1,:]:
                        f_write.write(' '.join([str(x_np) for x_np in item_np]) + '\n')
                    f_write.write(' '.join([str(x_np) for x_np in np_feature_to_store[-1,:] ]))
                    f_write.write(" ]\n")
                    if i % 100 == 0:
                        print("finished " + str(i) + " files")
                    i = i + 1 
                    


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
