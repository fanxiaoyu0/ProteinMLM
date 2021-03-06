from .core import SingletonMeta
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict,Tuple
import io
# import pandas as pd
from pathlib import Path
import uuid
import subprocess
import numpy as np
import string
import random
import argparse


logger = logging.getLogger(__name__)

class CommonVars(metaclass=SingletonMeta):
    def __init__(self, overrides: Dict[str, any]={}) -> None:
        self.app = os.getenv('APPNAME', 'PGEN')
        self.workspace = os.getenv(f'{self.app}_WORKSPACE', '/workspace')
        self.project_dir = f'{self.workspace}'
        self.data_dir = f'{self.workspace}/data'
        self.logs_dir = f'{self.workspace}/logs'

        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        self.__dict__.update(overrides)
        logger.debug(self.__repr__())
        
    def __repr__(self):
        return f"{self.__dict__}"

def get_common() -> CommonVars:
    return CommonVars()

def setup_logger(app_logger, output_dir="/workspace/logs", log_level=logging.INFO):
    standard_log_level = log_level
    app_logger.setLevel(standard_log_level)

    log_formatter = logging.Formatter('[%(levelname)s][%(name)s] %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    logging_stream_handler = logging.StreamHandler()
    logging_stream_handler.setLevel(standard_log_level)
    logging_stream_handler.setFormatter(log_formatter)
    app_logger.addHandler(logging_stream_handler)

    overrides = {} # {'workspace': 'test'}

    logging_file_handler = RotatingFileHandler(
        filename=f"{output_dir}/pgen.log", 
        maxBytes=2e6,
        backupCount=1)
    logging_file_handler.setLevel(standard_log_level)
    logging_file_handler.setFormatter(log_formatter)
    app_logger.addHandler(logging_file_handler)

class RawAndDefaultsFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass

def _open_if_is_name(filename_or_handle):
    out = filename_or_handle
    input_type = "handle"
    try:
        out = open(filename_or_handle,"r")
        input_type = "name"
    except TypeError:
        pass
    except Exception as e:
        raise(e)

    return (out, input_type)

LEGAL_AA_CODES={c:i for i,c in enumerate("-ACDEFGHIKLMNOPQRSTUVWYX")} #mapping characters to indexes
AA_REVERSE_LOOKUP=len(LEGAL_AA_CODES)*[""]
for i in LEGAL_AA_CODES:
    AA_REVERSE_LOOKUP[LEGAL_AA_CODES[i]] = i #mapping indexes to characters

def msa_to_matrix(list_of_strings):
    """converts an msa to a numerical array of dimensions num_sequences x alignment length"""
    msa = list_of_strings
    tr = LEGAL_AA_CODES
    out = np.zeros((len(msa), len(msa[0])), dtype=np.uint8)
    for i in range(len(msa)):
        for j in range(len(msa[0])):
            out[i,j] = tr[msa[i][j]]
    return out


def msa_to_frequencies(inp_names, inp_seqs, description_prefix=None):
    '''
        Converts a multiple sequence alignment (msa) (a list of sequences) into an array of dimensions number_of_legal_aa_characters x sequence_length
        where the values are in the range (0,1) and sum to 1 for each column

        description_prefix can be used to select a subset, based on name, of the sequences in the input msa.

        adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/correlated_mutations.py
    '''
    seqs = list()

    for i in range(len(inp_seqs)):
        if ( (description_prefix is None) or (inp_names[i].startswith(description_prefix)) ):
            seqs.append(inp_seqs[i])

    out = np.zeros((len(LEGAL_AA_CODES), len(inp_seqs[0]) ))
    
    msa_matrix = msa_to_matrix(seqs) # num_seqs x seq_len = AA_idx

    for i in range(msa_matrix.shape[0]):
        for pos in range(msa_matrix.shape[1]):
            out[msa_matrix[i,pos],pos] += 1
    return out.astype(np.float64) / np.float64(len(seqs))

def msa_to_second_order_statistics(inp_names, inp_seqs, description_prefix=None):
    '''
      calculates raw correlations between positions in an msa_array

      output: an array of dimensions (alignment_length, alignment_length, AA_CODES_length, AA_codes_length)
      where the indexes are: (first_position_AA, second_position_AA,first_position_index, second_position_index)
      and the values are the frequency of the associations (0-1)
        description_filter can be used to select a subset, based on name, of the sequences in the input msa.

    adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/correlated_mutations.py
    '''
    seqs = list()

    for i in range(len(inp_seqs)):
        if ( (description_prefix is None) or (inp_names[i].startswith(description_prefix)) ):
            seqs.append(inp_seqs[i])
    
    msa_matrix = msa_to_matrix(seqs) # num_seqs x seq_len = AA_idx

    out = np.zeros((len(LEGAL_AA_CODES),len(LEGAL_AA_CODES),msa_matrix.shape[1],msa_matrix.shape[1]), dtype=np.uint32)
    for i in range(msa_matrix.shape[0]):
      # print(seq_number)
      for pos1 in range(msa_matrix.shape[1]):
        for pos2 in range(msa_matrix.shape[1]):
          out[msa_matrix[i,pos1], msa_matrix[i,pos2], pos1, pos2] += 1 

    return out.astype(np.float64)/np.float64(len(seqs))

def second_order_correlations(fos, sos):
    """
        sos[aa1,aa2,pos1,pos2] - fos[aa1,pos1]*fos[aa2,pos2]
    """

    # TODO: try to vectorize this.
    out = np.zeros(sos.shape)

    for aa1 in range(sos.shape[0]):
        for aa2 in range(sos.shape[1]):
            for pos1 in range(sos.shape[2]):
                for pos2 in range(sos.shape[3]):
                    out[aa1,aa2,pos1,pos2] = sos[aa1,aa2,pos1,pos2] - (fos[aa1,pos1] * fos[aa2,pos2])
    return out

def flatten_second_order(sos):
    """
        sos will be symmetric, so we only need one hez
    """
    out = np.zeros(sos.shape[2]*sos.shape[2])
    return out

def unalign(sequence: str) -> Tuple[str,list]:
    """
        input:
            sequence: the starting sequence
                if 'unalign' then convert to upper, delete ".", "*", "-"

        output:
            cleaned_sequence: the cleaned sequence
            gap_mask: a list containing chars or None. The idea is that to get a sequence with gaps in the same places
    """
    upperified = sequence.upper()
    acceptable = string.ascii_uppercase
    cleaned_list = list()
    gap_mask = list()
    for c in upperified:
        if c in string.ascii_uppercase:
            cleaned_list.append(c)
            gap_mask.append(None)
        else:
            gap_mask.append(c)
    return "".join(cleaned_list), gap_mask

def add_gaps_back(sequence: str, gap_mask: list) -> str:
    """
        input:
            sequence: the cleaned sequence
            gap_mask: a list containing chars or None. The idea is that to get a sequence with gaps in the same places, you will pull a character from the sequence for every "None" in the mask, and otherwise pull the mask character.
        output:
            a string of size len(gap_mask) where None positions have been replaced, in order, by characters from sequence.

        example:
            add_gaps_back("MTGQ", [None,'-','-',None,None,".","-",None,"*"])
                = "M--TG.-Q*"
    """
    out = list()
    seq_index = 0
    for c in gap_mask:
        if c is None:
            out.append(sequence[seq_index])
            seq_index += 1
        else:
            out.append(c)
    return "".join(out)


def parse_fasta(filename, return_names=False, clean=None, full_name=False): 
    """
        adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/parsers.py
        

        input:
            filename: the name of a fasta file or a filehandle to a fasta file.
            return_names: if True then return two lists: (names, sequences), otherwise just return list of sequences
            clean: {None, 'upper', 'delete', 'unalign'}
                    if 'delete' then delete all lowercase "." and "*" characters. This is usually if the input is an a2m file and you don't want to preserve the original length.
                    if 'upper' then delete "*" characters, convert lowercase to upper case, and "." to "-"
                    if 'unalign' then convert to upper, delete ".", "*", "-"

        output: sequences or names, sequences
    """
    
    prev_len = 0
    prev_name = None
    prev_seq = ""
    out_seqs = list()
    out_names = list()
    (input_handle, input_type) = _open_if_is_name(filename)

    for line in input_handle:
        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] == ">":
            if full_name:
                name = line[1:]
            else:
                parts = line.split(None, 1)
                name = parts[0][1:]
            out_names.append(name)
            if (prev_name is not None):
                out_seqs.append(prev_seq)
            prev_len = 0
            prev_name = name
            prev_seq = ""
        else:
            prev_len += len(line)
            prev_seq += line
    if (prev_name != None):
        out_seqs.append(prev_seq)

    if input_type == "name":
        input_handle.close()
    
    if clean == 'delete':
        # uses code from: https://github.com/facebookresearch/esm/blob/master/examples/contact_prediction.ipynb
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        translation = str.maketrans(deletekeys)
        remove_insertions = lambda x: x.translate(translation)

        for i in range(len(out_seqs)):
            out_seqs[i] = remove_insertions(out_seqs[i])
    
    elif clean == 'upper':
        deletekeys = {'*': None, ".": "-"}
        translation = str.maketrans(deletekeys)
        remove_insertions = lambda x: x.translate(translation)

        for i in range(len(out_seqs)):
            out_seqs[i] = remove_insertions(out_seqs[i].upper())
    elif clean == 'unalign':
        deletekeys = {'*': None, ".": None, "-": None}
        
        translation = str.maketrans(deletekeys)
        remove_insertions = lambda x: x.translate(translation)
        
        for i in range(len(out_seqs)):
            out_seqs[i] = remove_insertions(out_seqs[i].upper())
    elif clean is not None:
        raise ValueError(f"unrecognized input for clean parameter: {clean}")

    if return_names:
        return out_names, out_seqs
    else:
        return out_seqs

def parse_fasta_string(fasta_string, return_names=False): 
    """
        adapted from: https://bitbucket.org/seanrjohnson/srj_chembiolib/src/master/parsers.py
        
        input: a fasta string
        output: a list of sequences from the fasta file
    """
    inp = io.StringIO(fasta_string)
    return parse_fasta(inp, return_names)


def write_sequential_fasta(path, sequences):
    """
        writes a fasta file to path, where the sequences are named as integers from 0 to len(sequences) - 1.
    """
    with open(path,"w") as fasta_out:
        for i, seq in enumerate(sequences):
            print(f">{i}\n{seq}",file=fasta_out)

def write_partitioned_fasta(path, sequences):
    """
        sequences is a dict where keys are categories of sequences and values are lists of sequences:

        writes a fasta file to path, where the sequences are named as category_[0-9]+ .
    """
    with open(path,"w") as fasta_out:
        for category, seqs in sequences.items():
            for i, seq in enumerate(seqs):
                print(f">{category}_{i}\n{seq}",file=fasta_out)

# def first_order_statistics(prefixes, names, seqs):
#     """
#         input: 
#             prefixes: a list of strings, such that each sequence name starts with one of the prefixes
#             names: list of sequence names
#             seqs: list of aligned sequences. Must all be of same length.
#         output: 
#             a data frame with index=['position','AA'], columns='prefix'
#             values are frequencies
#     """
    
#     out_dict = dict() #{position: {AA: {prefix: [count] }}}
#     prefix_counts = {x:0 for x in prefixes}
    
#     for i,name in enumerate(names):
#         prefix = None
#         for prfx in prefixes:
#             if name.startswith(prfx):
#                 prefix = prfx
#                 prefix_counts[prefix] += 1
#         if prefix is None:
#             raise(Exception(f"{name} does not have valid prefix"))
#         for pos_1,aa in enumerate(seqs[i]):
#             if pos_1 not in out_dict:
#                 out_dict[pos_1] = dict()
#             if aa not in out_dict[pos_1]:
#                 out_dict[pos_1][aa] = {prfx: 0 for prfx in prefixes}
#             out_dict[pos_1][aa][prefix] += 1
#     out_list = list()
    
#     for position, AAs in out_dict.items():
#         for AA, prefxes in AAs.items():
#             for prefix, count in prefxes.items():
#                 out_list.append({"position": position, "AA": AA, "prefix": prefix, "frequency": count/prefix_counts[prefix]})

#     df = pd.DataFrame(out_list)

#     return df.pivot(index=['position','AA'], columns='prefix')['frequency']

# def second_order_statistics(prefixes, names, seqs):
#     """
#         input: 
#             prefixes: a list of strings, such that each sequence name starts with one of the prefixes
#             names: list of sequence names
#             seqs: list of aligned sequences. Must all be of same length.
#         output: 
#             a data frame with index=['position_1', 'position_2', 'AA1', 'AA2'], columns='prefix'
#             values are frequencies
#     """
    
#     out_dict = dict() #{position: {AA: {prefix: [count] }}}
#     prefix_counts = {x:0 for x in prefixes}
    
#     for i,name in enumerate(names):
#         prefix = None
#         for prfx in prefixes:
#             if name.startswith(prfx):
#                 prefix = prfx
#                 prefix_counts[prefix] += 1
#         if prefix is None:
#             raise(Exception(f"{name} does not have valid prefix"))
#         for pos_1, aa1 in enumerate(seqs[i]):
            
#             if pos_1 not in out_dict:
#                 out_dict[pos_1] = dict()
#             if aa1 not in out_dict[pos_1]:
#                 out_dict[pos_1][aa1] = dict()
            
#             for pos_2 in range(pos_1+1, len(seqs[i])):
#                 aa2 = seqs[i][pos_2]
#                 if pos_2 not in out_dict[pos_1][aa1]:
#                     out_dict[pos_1][aa1][pos_2] = dict()
#                 if aa2 not in out_dict[pos_1][aa1][pos_2]:
#                     out_dict[pos_1][aa1][pos_2][aa2] = {prfx: 0 for prfx in prefixes}
#                 out_dict[pos_1][aa1][pos_2][aa2][prefix] += 1
                
            
#     out_list = list()
    
#     for position1, AA1s in out_dict.items():
#         for AA1, position2s in AA1s.items():
#             for position2, AA2s in position2s.items():
#                 for AA2, prefxes in AA2s.items():
#                     for prefix, count in prefxes.items():
#                         out_list.append({"position_1": position1, "position_2":position2, "AA1": AA1, "AA2": AA2, "prefix": prefix, "frequency": count/prefix_counts[prefix]})

#     df = pd.DataFrame(out_list)

#     return df.pivot(index=['position_1', 'position_2', 'AA1','AA2'], columns='prefix')['frequency']




# def third_order_statistics(prefixes, names, seqs):
#     """
#         input: 
#             prefixes: a list of strings, such that each sequence name starts with one of the prefixes
#             names: list of sequence names
#             seqs: list of aligned sequences. Must all be of same length.
#         output: 
#             a data frame with index=['position_1', 'position_2', 'position_3', 'AA1', 'AA2', 'AA3'], columns='prefix'
#             values are frequencies
#     """
    
#     out_dict = dict() #{position: {AA: {prefix: [count] }}}
#     prefix_counts = {x:0 for x in prefixes}
    
#     for i,name in enumerate(names):
#         prefix = None
#         for prfx in prefixes:
#             if name.startswith(prfx):
#                 prefix = prfx
#                 prefix_counts[prefix] += 1
#         if prefix is None:
#             raise(Exception(f"{name} does not have valid prefix"))
#         for pos_1, aa1 in enumerate(seqs[i]):
            
#             if pos_1 not in out_dict:
#                 out_dict[pos_1] = dict()
#             if aa1 not in out_dict[pos_1]:
#                 out_dict[pos_1][aa1] = dict()
            
#             for pos_2 in range(pos_1+1, len(seqs[i])):
#                 aa2 = seqs[i][pos_2]
#                 if pos_2 not in out_dict[pos_1][aa1]:
#                     out_dict[pos_1][aa1][pos_2] = dict()
#                 if aa2 not in out_dict[pos_1][aa1][pos_2]:
#                     out_dict[pos_1][aa1][pos_2][aa2] = dict()
                
#                 for pos_3 in range(pos_2+1,len(seqs[i])):
#                     aa3 = seqs[i][pos_3]
#                     if pos_3 not in out_dict[pos_1][aa1][pos_2][aa2]:
#                         out_dict[pos_1][aa1][pos_2][aa2][pos_3] = dict()
#                     if aa3 not in out_dict[pos_1][aa1][pos_2][aa2][pos_3]:
#                         out_dict[pos_1][aa1][pos_2][aa2][pos_3][aa3] = {prfx: 0 for prfx in prefixes}
#                     out_dict[pos_1][aa1][pos_2][aa2][pos_3][aa3][prefix] += 1
                
            
#     out_list = list()
#     #print(out_dict)
#     for position1, AA1s in out_dict.items():
#         for AA1, position2s in AA1s.items():
#             for position2, AA2s in position2s.items():
#                 for AA2, position3s in AA2s.items():
#                     for position3, AA3s in position3s.items():
#                         for AA3, prefxes in AA3s.items():
#                             for prefix, count in prefxes.items():
#                                 out_list.append({"position_1": position1, "position_2": position2, "position_3": position3, "AA1": AA1, "AA2": AA2, "AA3": AA3, "prefix": prefix, "frequency": count/prefix_counts[prefix]})

#     df = pd.DataFrame(out_list)

#     return df.pivot(index=['position_1', 'position_2', 'position_3', 'AA1', 'AA2', 'AA3'], columns='prefix')['frequency']

def generate_alignment(sequences, tmp_dir="/tmp"):
    """
        uses mafft to align sequences.
        
        sequences is a dict where keys are categories of sequences and values are lists of sequence
        
        
        returns (seq_names, sequences) as a tuple of lists.

    """

    #tmp_fasta_path = str((Path(tmp_dir) / str(uuid.uuid4())).with_suffix(".fasta"))
    tmp_fasta_path = str((Path(tmp_dir) / str(uuid.uuid4())).with_suffix(".fasta"))
    tmp_fasta_out_path = str((Path(tmp_dir) / str(uuid.uuid4())).with_suffix(".fasta"))
    write_partitioned_fasta(tmp_fasta_path,sequences)
    align_out = subprocess.run(['mafft', '--thread', '8', '--maxiterate', '1000', '--globalpair', tmp_fasta_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        align_out.check_returncode()
    except:
        print(align_out.stderr)
        raise(Exception)
    return parse_fasta_string(align_out.stdout.decode('utf-8'),True)

# def delete_msa_end_gaps(msa: list, gapchars: str = "-.") -> list:
#     """
        
#     """

#     first_non_gap_overall = len(msa[0])
#     last_non_gap_overall = 0
#     for s in range(len(msa)):
#         first_non_gap = None
#         last_non_gap = None
#         for i in range(len(msa[s])):
#             if msa[s][i] in gapchars:
#                 pass



class SequenceSubsetter:
    subset_strategies = {"random","in_order"}

    @classmethod
    def subset(cls, seq_list: list, n: int, keep_first: bool = False, strategy: str = "random", random_seed: int = None) -> list:
        """
            input:
                seq_list: a list of protein sequence strings
                n: how many members of seq_list to copy into the output. If n > len(seq_list), a copy of seq_list will be returned
                keep_first: if set, then copy seq_list[0] into output and sample n-1 additional items
                strategy: 
                    "random": take sequences randomly from seq_list (without replacement)
                    "in_order": take the top n sequences from seq_list
                random_seed: provided to the random number generator.
                
                Not implemented:
                delete_end_gaps: if supplied then truncates the alignment by deleting all positions before the first non-gap position in any sequnence and after the last non-gap position in any sequence.
        """
        #TODO: implement delete_end_gaps

        output = list()
        if n <= 0:
            return output

        tmp_list = seq_list.copy()
        if keep_first:
            output.append(seq_list[0])
            n = n - 1
            tmp_list = tmp_list[1:]

        if strategy not in cls.subset_strategies:
            raise ValueError(f"sampler strategy {strategy} not recognized, must be one of {cls.subset_strategies}")
        elif strategy == "random":
            random.Random(random_seed).shuffle(tmp_list)
            output += tmp_list[0:n]
        elif strategy == "in_order":
            output += tmp_list[0:n]
        
        # if delete_end_gaps:
        #     output = delete_msa_endgaps(output)
        return output
