import torch
import numpy as np
import os
from Modules.PointTransformerV3.PointTransformerV3 import PointTransformerWithHeads
from Modules.train_utils import run_training
from Modules.DataLoading.TreeSet import TreeSet, get_dataloader, get_treesets_plot_split, get_treesets_random_split
from Modules.Utils import EarlyStopper
from timm.scheduler.cosine_lr import CosineLRScheduler
import argparse
import sys
import fastprogress
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def setup_logging(model_save_path):
    # Create log file path by replacing `.pt` with `.log`
    log_file = model_save_path.replace('.pt', '.log') if model_save_path else 'training_log.log'

    # Set up logging configuration
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

    # Log the initial arguments
    logging.info("Training started with the following parameters:")

def log_parameters(args, test_plot=None):
    # Log the parameters used in the script
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Voxel size: {args.voxel_size}")
    logging.info(f"Early stopping patience: {args.patience_es}")
    logging.info(f"Warmup steps: {args.warmup_t}")
    logging.info(f"Minimum learning rate: {args.lr_min}")
    logging.info(f"Cosine initial t: {args.t_initial}")
    logging.info(f"Progress bar disabled: {args.no_progress_bar}")
    logging.info(f"Feature channels: {args.dim_feat}")
    logging.info(f"Use features:  {args.features}")
    logging.info(f"Noise Threshold: {args.noise_threshold}")
    logging.info(f"Model save path: {args.model_save_path}")
    logging.info(f"Used extra noise: {args.extra_noise}")
    logging.info(f"Semantic loss multiplier: {args.sem_loss_mult}")
    logging.info(f"Offset loss multiplier: {args.off_loss_mult}")
    logging.info(f"Cross validation: {args.cross_validate}")
    if test_plot:
        logging.info(f"Test plot: {test_plot}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train TreeLearn model with custom parameters.")
    
    # Define command-line arguments
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the optimizer")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size for the model")
    parser.add_argument("--patience_es", type=int, default=25, help="Patience for early stopping")
    parser.add_argument("--warmup_t", type=int, default=20, help="Warmup steps for the scheduler")
    parser.add_argument("--lr_min", type=float, default=0.0001, help="Minimum learning rate for the scheduler")
    parser.add_argument("--no_progress_bar", action="store_true", help="Disable the progress bar but keep logs")
    parser.add_argument("--dim_feat", type=int, default=1, help="The number of feature channels")
    parser.add_argument("--features", action="store_true", help="Whether to use features for training")
    parser.add_argument("--t_initial", type=int, default=50, help="The number of epochs after which the learning rate for cosine is reseted")
    parser.add_argument("--model_save_path", type=str, default=None, help="The path to which the model is saved")
    parser.add_argument("--noise_threshold", default=0.1, type=float, help="The threshold offset label length for training")
    parser.add_argument("--verbose", type=bool, default=False, help="Whether to print messages during training")
    parser.add_argument("--extra_noise", action="store_true", help="Use extra dataset for noise prediction")
    parser.add_argument("--sem_loss_mult", type=float, default=1.0, help="Weighting factor of semantic loss")
    parser.add_argument("--off_loss_mult", type=float, default=1.0, help="Weighting factor of offset loss")
    parser.add_argument("--cross_validate", action="store_true", help="Train models for cross validation across plots")
    parser.add_argument("--test_plots", type=int, nargs='+', help="The plots that should be used as test_plots", default=[3,4,6,8])

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Setup logging
    setup_logging(args.model_save_path)
    
    ###### Define parameters #######
    batch_size = args.batch_size
    epochs = args.epochs

    # Train & Val loader
    data_root = os.path.join('data', 'labeled', 'offset')
    if args.extra_noise:
        data_root = os.path.join('data', 'labeled', 'noise')
    # Train & Val loader
    train_root = os.path.join('data', 'labeled', 'offset', 'trainset')
    val_root = os.path.join('data', 'labeled', 'offset', 'testset')
    if args.extra_noise:
        train_root = os.path.join('data', 'labeled', 'noise', 'trainset')
        val_root = os.path.join('data', 'labeled', 'noise', 'testset')

    ####### CROSS VALIDATED PART ########
    if args.cross_validate:
        for num in args.test_plots:
            # Place the models in a folder with the original model name. Not the best code, but then this works
            # interchangably with the non-cross-validated version
            filename = os.path.splitext(os.path.basename( args.model_save_path ))[0]
            new_save_path = args.model_save_path.replace(f'{filename}.pt', f'{filename}_CV/{filename}_P{num}.pt')
            new_dict = args.model_save_path.replace(f'{filename}.pt', f'{filename}_CV/')
            
            if not os.path.exists( new_dict ):
                os.makedirs( new_dict )
            

            setup_logging(new_save_path)
            log_parameters(args, num)

            trainset, valset = get_treesets_plot_split(data_root, test_plot=num, noise_distance=args.noise_threshold)

            print(f"\nTesting on plot number {num}\nTrainset len: {len(trainset)}\tValset len: {len(valset)}")

            train_loader = get_dataloader(trainset, batch_size, num_workers=0, training=True, collate_fn=trainset.collate_fn_voxel)
            val_loader = get_dataloader(valset, batch_size, num_workers=0, training=False, collate_fn=valset.collate_fn_voxel)

            # spatial shape =  [30m,30m,50m], depends on voxel size
            # spatial_shape = [ 
            #     np.round( args.spatial_shape[0]/args.voxel_size ).astype(int), 
            #     np.round( args.spatial_shape[1]/args.voxel_size ).astype(int), 
            #     np.round( args.spatial_shape[2]/args.voxel_size ).astype(int) 
            # ]

            # Model
            model = PointTransformerWithHeads(
                dim_feat=args.dim_feat, use_feats=args.features, voxel_size=args.voxel_size, 
                loss_multiplier_semantic=args.sem_loss_mult, loss_multiplier_offset=args.off_loss_mult
                ).cuda()

            # Scheduler and optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=args.t_initial,
                lr_min=args.lr_min,
                cycle_decay=1,
                warmup_lr_init=0.00001,
                warmup_t=args.warmup_t,
                cycle_limit=1,
                t_in_epochs=True,
            )

            # Early stopper
            early_stopper = EarlyStopper(verbose=args.verbose, patience=args.patience_es, model_save_path=new_save_path)

            # Control over progress bar
            if args.no_progress_bar:
                fastprogress.fastprogress.NO_BAR = True  # Suppress the progress bar
            else:
                fastprogress.fastprogress.NO_BAR = False  # Enable the progress bar


            # Start training
            run_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                epochs=epochs,
                scheduler=scheduler,
                early_stopper=early_stopper,
                verbose=args.verbose
            )

    ####### RANDOM TESTSET PART #########
    else:    

        # Setup logging
        setup_logging(args.model_save_path)
        
        # Log parameters
        log_parameters(args)

        trainset, valset = get_treesets_random_split(data_root, noise_distance=args.noise_threshold)

        train_loader = get_dataloader(trainset, batch_size, num_workers=0, training=True, collate_fn=trainset.collate_fn_voxel)
        val_loader = get_dataloader(valset, batch_size, num_workers=0, training=False, collate_fn=valset.collate_fn_voxel)

        # Model
        model = PointTransformerWithHeads(
            dim_feat=args.dim_feat, use_feats=args.features, voxel_size=args.voxel_size, 
            loss_multiplier_semantic=args.sem_loss_mult, loss_multiplier_offset=args.off_loss_mult
            ).cuda()

        # Scheduler and optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=args.t_initial,
            lr_min=args.lr_min,
            cycle_decay=1,
            warmup_lr_init=0.00001,
            warmup_t=args.warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
        )

        # Early stopper
        early_stopper = EarlyStopper(verbose=args.verbose, patience=args.patience_es, model_save_path=args.model_save_path)

        # Control over progress bar
        if args.no_progress_bar:
            fastprogress.fastprogress.NO_BAR = True  # Suppress the progress bar
        else:
            fastprogress.fastprogress.NO_BAR = False  # Enable the progress bar


        # Start training
        run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epochs=epochs,
            scheduler=scheduler,
            early_stopper=early_stopper,
            verbose=args.verbose
        )