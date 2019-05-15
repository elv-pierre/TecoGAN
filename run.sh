#!/bin/bash
cudaID=$1
vidID=$2
PYTHON=/home/pierre.gleize/anaconda3/envs/sr/bin/python
SF=$3
sudo $PYTHON main.py --cudaID $cudaID --output_dir ./results/ --summary_dir ./results/log/ --mode inference --input_dir_LR ./LR/$vidID --output_pre $vidID --num_resblock 16 --checkpoint ./model/TecoGAN --output_ext png --starting_frame $SF
