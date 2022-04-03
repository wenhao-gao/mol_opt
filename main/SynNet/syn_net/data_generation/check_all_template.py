"""
This file checks if a set of reactions are represented by a set of reaction 
templates. Originally written by Jake. Wenhao edited.
"""
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from rdkit import RDLogger


def split_rxn_parts(rxn):
    '''
    Given SMILES reaction, splits into reactants, agents, and products

    Args:
        rxn (str): SMILES-encoded reaction.

    Returns:
        list: Contains sets of reactants, agents, and products as RDKit molecules.
    '''
    rxn_parts     = rxn.strip().split('>')
    rxn_reactants = set(rxn_parts[0].split('.'))
    rxn_agents    = None if not rxn_parts[1] else set(rxn_parts[1].split('.'))
    rxn_products  = set(rxn_parts[2].split('.'))

    reactants, agents, products = set(), set(), set()

    # convert reactants to rdkit molecules
    for r in rxn_reactants:
        reactants.add(Chem.MolFromSmiles(r))

    # if present, convert agents to rdkit molecules
    if rxn_agents:
        for a in rxn_agents:
            agents.add(Chem.MolFromSmiles(a))

    # convert products to rdkit molecules
    for p in rxn_products:
        products.add(Chem.MolFromSmiles(p))

    return [reactants, agents, products]


def rxn_template(rxn_smiles, templates):
    '''
    Given a reaction, checks whether it matches any templates.

    Args:
        rxn_smiles (str): Reaction in Reaction SMILES format.
        templates (dict): Maps RDKit reactions to template names.

    Returns:
        str: Matching template name. If no templates matched, returns None.
    '''
    rxn_parts = split_rxn_parts(rxn_smiles)
    reactants, agents, products = rxn_parts[0], rxn_parts[1], rxn_parts[2]
    temp_match = None

    for t in templates:
        agents_match = None
        products_match = None

        # check whether all reactants match template
        reactants_match = True
        for r in reactants:
            if not t.IsMoleculeReactant(r):
                reactants_match = False

        # if reactants matched, check whether all agents match template
        if reactants_match:
            agents_match = True
            for a in agents:
                if not t.IsMoleculeAgent(a):
                    agents_match = False

        # if reactants and agents matched, check whether all products match template
        if agents_match:
            products_match = True
            for p in products:
                if not t.IsMoleculeProduct(p):
                    products_match = False

        # if reactants, agents, and products match template, add template to matches
        if products_match:
            temp_match = t

    if not temp_match:
        return temp_match

    # get matching template names
    return templates[temp_match]


def route_templates(route, templates):
    '''
    Given synthesis route, checks whether all reaction steps are in template list

    Args:
        route (list): Contains reaction steps (str Reaction SMILES).
        templates (dict): Maps RDKit reactions to template names.

    Returns:
        List of matching template names (as strings). If no templates matched,
            returns empty list.
    '''
    synth_route = []
    tree_match = True
    for rxn_step in route:
        res = rxn_template(rxn_step, templates)
        if not res:
            tree_match = False
            synth_route = []
            break
        else:
            synth_route.append(res)

    return synth_route

if __name__ == '__main__':

    disable_RDLogger = True  # disables RDKit warnings
    if disable_RDLogger:
        RDLogger.DisableLog('rdApp.*')

    rxn_set_path = '/path/to/rxn_set.txt'

    rxn_set = open(rxn_set_path, 'r')
    templates = {}

    for rxn in rxn_set:
        rxn_name  = rxn.split('|')[0]
        template  =  rxn.split('|')[1].strip()
        rdkit_rxn = AllChem.ReactionFromSmarts(template)
        rdChemReactions.ChemicalReaction.Initialize(rdkit_rxn)
        templates[rdkit_rxn] = rxn_name

    rxn_smiles = 'ClCC1CO1.NC(=O)Cc1ccc(O)cc1>>NC(=O)Cc1ccc(OCC2CO2)cc1'
    print(rxn_smiles)
    print(rxn_template(rxn_smiles, templates))
    print('------------------------------------------------------')
    synthesis_route = [
        'C(CCc1ccccc1)N(Cc1ccccc1)CC(O)c1ccc(O)c(C(N)=O)c1>>CC(CCc1ccccc1)NCC(O)c1ccc(O)c(C(N)=O)c1',
        'CC(CCc1ccccc1)N(CC(=O)c1ccc(O)c(C(N)=O)c1)Cc1ccccc1>>CC(CCc1ccccc1)N(Cc1ccccc1)CC(O)c1ccc(O)c(C(N)=O)c1',
        'CC(CCc1ccccc1)NCc1ccccc1.NC(=O)c1cc(C(=O)CBr)ccc1O>>CC(CCc1ccccc1)N(CC(=O)c1ccc(O)c(C(N)=O)c1)Cc1ccccc1'
    ]
    print(synthesis_route)
    print(route_templates(synthesis_route, templates))
