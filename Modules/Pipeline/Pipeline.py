import os
import argparse

from .QSMFitting import fitQSM
from .SuperSampling import superSample
from .TreeLearnPredicting import makePredictions


def run_pipeline( input_dir, output_dir, denoise ):

    print("Loading input data...")
    cloud_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]

    print("Making predictions...")
    makePredictions( cloud_list, output_dir, denoise=denoise )

    # load predicted data

    if denoise:
        cloud_list = [os.path.join( output_dir, f ) for f in os.listdir(output_dir) if f.endswith("pred_denoised.txt")]
    else:
        cloud_list = [os.path.join( output_dir, f ) for f in os.listdir(output_dir) if f.endswith("pred.txt")]

    print("Super sampling... ")
    superSample( cloud_list, output_dir, min_height=0 )

    # load super sampled data
    if denoise:
        cloud_list = [os.path.join( output_dir, f ) for f in os.listdir(output_dir) if f.endswith("denoised_supsamp.txt")]
    else:
        cloud_list = [os.path.join( output_dir, f ) for f in os.listdir(output_dir) if f.endswith("pred_supsamp.txt")]

    print("QSM Fitting...")
    fitQSM( cloud_list, output_dir )