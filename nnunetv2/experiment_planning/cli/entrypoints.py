from .argument_parsers import (
    create_fingerprint_parser,
    create_planning_parser,
    create_preprocessing_parser,
    create_combined_parser,
    get_default_num_processes
)
from ..core.fingerprint_extraction import extract_fingerprints
from ..core.experiment_planning import plan_experiments
from ..core.preprocessing_coordination import preprocess


def extract_fingerprint_entry():
    """Entry point for dataset fingerprint extraction."""
    parser = create_fingerprint_parser()
    args, unrecognized_args = parser.parse_known_args()
    extract_fingerprints(args.d, args.fpe, args.np, args.verify_dataset_integrity, args.clean, args.verbose)


def plan_experiment_entry():
    """Entry point for experiment planning."""
    parser = create_planning_parser()
    args, unrecognized_args = parser.parse_known_args()
    plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name, args.overwrite_target_spacing,
                     args.overwrite_plans_name)


def preprocess_entry():
    """Entry point for preprocessing."""
    parser = create_preprocessing_parser()
    args, unrecognized_args = parser.parse_known_args()

    if args.np is None:
        np = get_default_num_processes(args.c)
    else:
        np = args.np

    preprocess(args.d, args.plans_name, configurations=args.c, num_processes=np, verbose=args.verbose)


def plan_and_preprocess_entry():
    """Entry point for combined planning and preprocessing."""
    parser = create_combined_parser()
    args = parser.parse_args()

    # fingerprint extraction
    print("Fingerprint extraction...")
    extract_fingerprints(args.d, args.fpe, args.npfp, args.verify_dataset_integrity, args.clean, args.verbose)

    # experiment planning
    print('Experiment planning...')
    plans_identifier = plan_experiments(args.d, args.pl, args.gpu_memory_target, args.preprocessor_name,
                                        args.overwrite_target_spacing, args.overwrite_plans_name)

    # manage default np
    if args.np is None:
        np = get_default_num_processes(args.c)
    else:
        np = args.np

    # preprocessing
    if not args.no_pp:
        print('Preprocessing...')
        preprocess(args.d, plans_identifier, args.c, np, args.verbose)


if __name__ == '__main__':
    plan_and_preprocess_entry()