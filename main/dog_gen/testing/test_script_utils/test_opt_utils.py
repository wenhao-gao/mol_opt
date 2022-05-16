
from syn_dags.script_utils import opt_utils

def test_gaucamol_scorers():
    # These scores are taken from some of the runs with the baselines. We test that our interface obtains the same scores
    # This is mostly important as Deco Hop and Scaffold Hop changed names at some point (in the Guacamol code)
    # so running on different versions of Guacamol would obtain erroneous results.
    guac_name_mol_score = [
        ('guac_Perindopril_MPO','CCCC(NC(C)C(=O)n1nc(C2CCCC2)c2sn(C3CC3C(=O)O)c21)C(=O)OCC', 0.7643025682552586),
        ('guac_Amlodipine_MPO','CCOC(=O)C1=C(CN)NC(C)=C(C(=O)OCCOCc2nn(C)s2)C1c1ccccc1Cl', 0.8246211251235321),
        ('guac_Sitagliptin_MPO', 'CON=C(NN=Cc1ccc(N2CC(F)(F)N2)cc1)c1ccc(F)cn1', 0.6606539012699573),
        ('guac_Aripiprazole_similarity', 'O=C1NCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2O1',	0.9866666666666666),
        ('guac_Osimertinib_MPO', 'CSc1cnccc1NC(=O)c1nc(-c2ccc([PH](N)(O)N3CCN(C)CC3)cc2)ccc1[BiH+79]',	0.8555261858712448),
        ('guac_Ranolazine_MPO', 'Cc1c(COCCCCCCC(=O)OC(C)C)cccc1NC(=O)CCCCCCC(O)COc1ccccc1F',	0.9039763229494032),
        ('guac_Zaleplon_MPO', 'CC(=O)N(Cc1cc(-c2cccnc2)ccn1)c1cccc(O)c1',	0.6407232755171874),
        ('guac_Valsartan_SMARTS', 'O=C1COC(=O)N1Cc1ccc(-c2ccc(F)cc2)cc1Cl',	6.304158683343028e-08),
        ('guac_decoration_hop', 'COc1cc2ncnc(Nc3cccc4cnccc34)c2cc1OC',	0.9036551295980133),
        ('guac_scaffold_hop', 'C=S(=O)(c1ccc(C)cc1)n1ccc2c(Nc3ccc4nnsc4c3)nc(Cl)nc21',	0.5757575757575757),
        ('guac_Celecoxib_rediscovery', 'Cc1ccc(C=Nc2ccc(S(N)(=O)=O)cc2)cc1',	0.4588235294117647),
        ('guac_Troglitazone_rediscovery', 'Cc1c(C)c2c(c(C)c1O)CCC(C)(C(=O)NCOc1ccc(O)cc1)O2',	0.5094339622641509),
        ('guac_Albuterol_similarity', 'CCC(O)c1ccc(O)c(C#C[Ge](C)(O)Br)c1',	0.7719298245614035),
        ('guac_Fexofenadine_MPO', 'COC(=O)C1=CC(F)=CC=S1NC(=O)COCC(O)N1CCC(C(c2ccccc2)c2ccccc2)CC1',	0.784002259753152),
    ]

    for task_name, smiles, score in guac_name_mol_score:
        prop_eval = opt_utils.get_task(task_name)
        obtained_score = prop_eval.evaluate_molecules([smiles])[0]
        assert obtained_score == score
