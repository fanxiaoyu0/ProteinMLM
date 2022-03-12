import esm

# based on examples from here: https://github.com/facebookresearch/esm/blob/master/examples/contact_prediction.ipynb
class ESM1b():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()

class ESM6():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm1_t6_43M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()

class ESM12():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(
            "/home/protein_gibbs_sampler/model/ESM12/2/esm1_t12_85M_UR50S/2022-03-06-18-23-39/checkpoints/best-model-checkpoint-Mar06_18-23-40_ab04307d.pt")
        # self.model, self.alphabet = esm.pretrained.esm1_t12_85M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()

class ESM34():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm1_t34_670M_UR50S()
        self.batch_converter = self.alphabet.get_batch_converter()

class ESM_MSA1():
    def __init__(self):
        self.model, self.alphabet = esm.pretrained.esm_msa1_t12_100M_UR50S()
        # self.msa_transformer = model.eval().cuda()
        self.batch_converter = self.alphabet.get_batch_converter()