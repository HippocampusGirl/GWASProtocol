import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import multiprocessing as mp

from gwas.plots.helpers import filter_rois, check_existing_files, generate_and_save_manhattan_plot, save_dataframe_as_pickle
from gwas.plots.worker import create_dataframe_all_chr

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments"""
    parser = argparse.ArgumentParser(description="Generate Manhattan & QQ Plots")
    parser.add_argument(
        "--metadata_directory",
        help="Insert path to the metadata directory",
        required=True,
    )
    parser.add_argument(
        "--scoredata_directory",
        help="Insert path to the scoredata directory",
        required=True,
    )
    parser.add_argument(
        "--csvroi_file",
        help="Insert path to the csv rois file",
        default=None,
    )
    parser.add_argument(
        "output_directory",
        help='Insert path to output directory for dataframes and plots',
        required=True,
    )
    parser.add_argument(
        "--log-level", choices=logging.getLevelNamesMapping().keys(), default="INFO"
    )
    parser.add_argument("--num-threads", type=int, default=mp.cpu_count())
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    from gwas.log import logger, setup_logging

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    setup_logging(level=args.log_level, log_path=output_directory)

    labels_list = []
    if args.csvroi_file:
        labels_list = filter_rois(csv_path=args.csvroi_file)

    # Implement logic to work with nonexistent region of interest files

    # maybe confusing to use progressbar for labels and then not for chromomosomes but variant data for each chromosome
    
    for label in tqdm(range(len(labels_list)),desc="Processing labels"):
        if check_existing_files(directory=output_directory, label=label):
            continue
        cur_df = create_dataframe_all_chr(
            score_path=args.scoredata_directory, 
            metadata_path=args.metadata_directory, 
            label=label, cpu_count=args.num_threads
            )
        save_dataframe_as_pickle(dataframe=cur_df, directory=output_directory, label=label)
        
        generate_and_save_manhattan_plot(dataframe=cur_df, directory=output_directory,label=label)

if __name__ == '__main__':
    main()


