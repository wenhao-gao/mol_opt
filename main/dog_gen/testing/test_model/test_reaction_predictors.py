

import requests
import pytest
import multiset

from syn_dags.model import reaction_predictors


def test_open_nmt_run_list_of_reactant_sets_output(monkeypatch):
    def mock_post(*args, **kwargs):
        class MockResponse:
            @staticmethod
            def json():
                return [[{"n_best":1,"pred_score":-0.0002288818359375,
                          "src":"C [S-] . [Mg+] c 1 c c c ( Cl ) c c 1",
                          "tgt":"C S c 1 c c c ( Cl ) c c 1"},
                         {"n_best":1,"pred_score":-0.004589080810546875,
                          "src":"C C O C ( = O ) C 1 C C N ( C ( = O ) O C ( C ) ( C ) C ) C C 1 . C C ( C ) ( C ) O C ( = O ) N 1 C C N C C 1",
                          "tgt":"C C ( C ) ( C ) O C ( = O ) N 1 C C N ( C ( = O ) C 2 C C N ( C ( = O ) O C ( C ) ( C ) C ) C C 2 ) C C 1"}]]
            def raise_for_status(self):
                pass
        return MockResponse()

    monkeypatch.setattr(requests, "post", mock_post)
    nmt_pred = reaction_predictors.OpenNMTServerPredictor()
    out = nmt_pred._run_list_of_reactant_sets([
        multiset.FrozenMultiset(['C[S-]', '[Mg+]c1ccc(Cl)cc1']),
        multiset.FrozenMultiset(['CCOC(=O)C1CCN(C(=O)OC(C)(C)C)CC1', 'CC(C)(C)OC(=O)N1CCNCC1'])
    ])
    assert out[0] == ['CSc1ccc(Cl)cc1']
    assert out[1] == ['CC(C)(C)OC(=O)N1CCC(C(=O)N2CCN(C(=O)OC(C)(C)C)CC2)CC1'] # nb note canconcalised.


def test_open_nmt_run_list_of_reactant_sets_input(monkeypatch):
    def mock_post(add, data, **kwargs):

        assert data == '[{"src": "C [S-] . [Mg+] c 1 c c c ( Cl ) c c 1", "id": 0}, {"src": "C C O C ( = O ) C 1 C C N ( C ( = O ) O C ( C ) ( C ) C ) C C 1 . C C ( C ) ( C ) O C ( = O ) N 1 C C N C C 1", "id": 0}]'

        class MockResponse:
            @staticmethod
            def json():
                return [[{"n_best":1,"pred_score":-0.0002288818359375,
                          "src":"C [S-] . [Mg+] c 1 c c c ( Cl ) c c 1",
                          "tgt":"C S c 1 c c c ( Cl ) c c 1"},
                         {"n_best":1,"pred_score":-0.004589080810546875,
                          "src":"C C O C ( = O ) C 1 C C N ( C ( = O ) O C ( C ) ( C ) C ) C C 1 . C C ( C ) ( C ) O C ( = O ) N 1 C C N C C 1",
                          "tgt":"C C ( C ) ( C ) O C ( = O ) N 1 C C N ( C ( = O ) C 2 C C N ( C ( = O ) O C ( C ) ( C ) C ) C C 2 ) C C 1"}]]
            def raise_for_status(self):
                pass
        return MockResponse()

    monkeypatch.setattr(requests, "post", mock_post)
    nmt_pred = reaction_predictors.OpenNMTServerPredictor()
    out = nmt_pred._run_list_of_reactant_sets([
        multiset.FrozenMultiset(['C[S-]', '[Mg+]c1ccc(Cl)cc1']),
        multiset.FrozenMultiset(['CCOC(=O)C1CCN(C(=O)OC(C)(C)C)CC1', 'CC(C)(C)OC(=O)N1CCNCC1'])
    ])
