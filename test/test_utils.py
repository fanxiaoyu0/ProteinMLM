import pytest
from pgen import utils
import io

####### Fixtures #######
@pytest.fixture(scope="function")
def a2m_file():
    f = io.StringIO(
"""
>seq_1
mdgtrtsldieeysdtevqknqvlTLEEWQDKWVNGKTAFHQEQGHQLLKKHLDTflKGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYSeep*
>seq_2
........................TLEEWQDKWVNGKTAFHQEQGHQLLKKHLDT..KGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYS...*
>seq_3
mdgtrtsldieeysdtevqknqvlTLEEWQDKWVNGK
TAFHQEQGHQLLKKHLDTflKGKSGLRVFFPLCGKAV
EMKWFADRGHSVVGVEISELGIQEFFTEQNLSYSeep*

"""
    )
    return f

###### Tests #######

@pytest.mark.parametrize("test_input,expected", [
                                                ({'seq_list': [0,1,2,3,4,5], 'n':1, 'keep_first': True, 'strategy': "in_order"}, [0]),
                                                ({'seq_list': [0,1,2,3,4,5], 'n':1, 'keep_first': False, 'strategy': "in_order"}, [0]),
                                                ({'seq_list': [0,1,2,3,4,5], 'n':1, 'keep_first': True, 'strategy': "random"}, [0]),
                                                ({'seq_list': [0,1,2,3,4,5], 'n':3, 'keep_first': True, 'strategy': "in_order"}, [0,1,2]),
                                                ({'seq_list': [0,1,2,3,4,5], 'n':3, 'keep_first': False, 'strategy': "in_order"}, [0,1,2]),
                                                ({'seq_list': [0,1,2,3,4,5], 'n':0, 'keep_first': False, 'strategy': "in_order"}, []),
                                                ({'seq_list': [0,1,2,3,4,5], 'n':0, 'keep_first': True, 'strategy': "in_order"}, []),
                                                ({'seq_list': [0,1,2,3,4,5], 'n':5000, 'keep_first': True, 'strategy': "in_order"}, [0,1,2,3,4,5]),
                                                ({'seq_list': [0,1,2,3,4,5], 'n':5000, 'keep_first': False, 'strategy': "in_order"}, [0,1,2,3,4,5]),
                                                ])
def test_subsetter_1(test_input, expected):
    assert utils.SequenceSubsetter.subset(**test_input) == expected

def test_subsetter_2():
    test_input = {'seq_list': [0,1,2,3,4,5], 'n':5000, 'keep_first': True, 'strategy': "random", "random_seed": 1}
    output = utils.SequenceSubsetter.subset(**test_input)
    assert output[0] == 0
    assert set(test_input['seq_list']) == set(output)
    assert test_input['seq_list'] != output

def test_parse_fasta_1(a2m_file):
    names, sequences = utils.parse_fasta(a2m_file, return_names = True)
    assert names == ["seq_1","seq_2","seq_3"]
    assert sequences == ["mdgtrtsldieeysdtevqknqvlTLEEWQDKWVNGKTAFHQEQGHQLLKKHLDTflKGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYSeep*", 
                         "........................TLEEWQDKWVNGKTAFHQEQGHQLLKKHLDT..KGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYS...*",
                         "mdgtrtsldieeysdtevqknqvlTLEEWQDKWVNGKTAFHQEQGHQLLKKHLDTflKGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYSeep*"]

def test_parse_fasta_2(a2m_file):
    sequences = utils.parse_fasta(a2m_file, return_names = False, clean="delete")
    assert sequences == ["TLEEWQDKWVNGKTAFHQEQGHQLLKKHLDTKGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYS", 
                         "TLEEWQDKWVNGKTAFHQEQGHQLLKKHLDTKGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYS",
                         "TLEEWQDKWVNGKTAFHQEQGHQLLKKHLDTKGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYS"]


def test_parse_fasta_3(a2m_file):
    sequences = utils.parse_fasta(a2m_file, return_names = False, clean="upper")
    assert sequences == ['MDGTRTSLDIEEYSDTEVQKNQVLTLEEWQDKWVNGKTAFHQEQGHQLLKKHLDTFLKGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYSEEP', 
                         "------------------------TLEEWQDKWVNGKTAFHQEQGHQLLKKHLDT--KGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYS---",
                         'MDGTRTSLDIEEYSDTEVQKNQVLTLEEWQDKWVNGKTAFHQEQGHQLLKKHLDTFLKGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYSEEP'
                        ]

def test_parse_fasta_3(a2m_file):
    sequences = utils.parse_fasta(a2m_file, return_names = False, clean="unalign")
    assert sequences == ['MDGTRTSLDIEEYSDTEVQKNQVLTLEEWQDKWVNGKTAFHQEQGHQLLKKHLDTFLKGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYSEEP', 
                         "TLEEWQDKWVNGKTAFHQEQGHQLLKKHLDTKGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYS",
                         'MDGTRTSLDIEEYSDTEVQKNQVLTLEEWQDKWVNGKTAFHQEQGHQLLKKHLDTFLKGKSGLRVFFPLCGKAVEMKWFADRGHSVVGVEISELGIQEFFTEQNLSYSEEP'
                        ]


@pytest.mark.parametrize("test_input,expected", [
    (".*-ABCDE.*-", ("ABCDE",['.','*','-',None,None,None,None,None,'.','*','-'])),
    ("AB.*-AB", ("ABAB",[None,None,'.','*','-',None,None])),
    ])
def test_unalign_1(test_input,expected):
    assert utils.unalign(test_input) == expected

@pytest.mark.parametrize("test_input,expected", [
    (("ABCDE",['.','*','-',None,None,None,None,None,'.','*','-']),".*-ABCDE.*-"),
    (("ABAB",[None,None,'.','*','-',None,None]),"AB.*-AB"),
    (("MTGQ", [None,'-','-',None,None,".","-",None,"*"]),"M--TG.-Q*"),
    ])
def test_add_gaps_back(test_input,expected):
    assert utils.add_gaps_back(*test_input) == expected
