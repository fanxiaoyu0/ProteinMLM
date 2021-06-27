import argparse
from pgen.esm_msa_sampler import ESM_MSA_sampler
from pgen import models
from pgen.utils import write_sequential_fasta, parse_fasta, SequenceSubsetter
from pathlib import Path
import sys
from tqdm import trange
import math

model_map = {"esm_msa1":models.ESM_MSA1}


def main(input_h, output_p, args):
    clean_flag = 'upper'
    if args.delete_insertions:
        clean_flag = 'delete'

    gibbs_sampler = ESM_MSA_sampler(model_map[args.model](), device=args.device)


    with open(output_p / "specification.tsv","w") as output_h:
        
        for line in input_h:
            line = line.strip().split("\t")
            if len(line) == 3:
                print("\t".join(line))
                print("\t".join(line), file=output_h)
                name = line[0]
                line_args = eval(line[1])

                input_msa = parse_fasta(line[2], clean=clean_flag)

                batches = math.ceil(args.num_output_sequences / args.alignment_size )
                sequences = list()
                for i in trange(batches):
                    batch_msa = SequenceSubsetter.subset(input_msa, args.alignment_size, args.keep_first_sequence, args.subset_strategy)
                    sequences += gibbs_sampler.generate(n_samples=len(batch_msa), seed_msa=batch_msa, batch_size=args.batch_size, show_progress_bar=False, **line_args)
                write_sequential_fasta( output_p / (name + ".fasta"), sequences[0:args.num_output_sequences] )
            else:
                print(f"Expected 3 values in specification file (name, line_args, input_msa), got {len(line)}")
                print("\t".join(line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", default=".", help="a directory to save the outputs in")
    parser.add_argument("-i", default=None, help="tab separated file where the columns are as follows: [sample name] \\t [dict of arguments for the sampler] \\t [path to seed msa in fasta or a2m format].")
    parser.add_argument("--batch_size", type=int, default=1, required=True, choices={1,}, help="batch size for sampling (msa instances per iteration). Must be 1. This might change in the future.")
    parser.add_argument("--num_output_sequences", type=int, default=1, required=True, help="total number of sequences to generate.")
    parser.add_argument("--device", type=str, default="cpu", choices={"cpu","gpu"}, required=True, help="cpu or gpu")
    parser.add_argument("--model", type=str, default="esm_msa1", choices={"esm_msa1"}, help="which model to use")
    parser.add_argument("--delete_insertions", action='store_true', default=False, help="If set, then remove all lowercase and '.' characters from input sequences. Default: convert lower to upper and '.' to '-'.") #might want to have the option to keep "." in the msa and convert lower to upper (which would be consistent with the vocabulary, which has ".", but does not have lowercase characters.)

    parser.add_argument("--alignment_size", type=int, default=sys.maxsize, help="Sample this many sequences from the input alignment before doing gibbs sampling, recommended values are 32-256. Default: the entire input alignment.")

    parser.add_argument("--keep_first_sequence", action='store_true', default=False, help="If set, then keep the first sequence and sample the rest according to subset_strategy.")
    parser.add_argument("--subset_strategy", default="random", choices=SequenceSubsetter.subset_strategies, help="How to subset the input alignment to get it to the desired size.")

    args = parser.parse_args()

    if args.i is not None:
        input_handle=open(args.i, "r")
    else:
        input_handle = sys.stdin

    output_path = Path(args.o)

    output_path.mkdir(exist_ok=True)

    main(input_handle, output_path, args)

    if args.i is not None:
        input_handle.close()