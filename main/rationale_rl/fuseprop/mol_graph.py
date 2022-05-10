import torch
import random
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from fuseprop.chemutils import *
from fuseprop.nnutils import *
from fuseprop.vocab import common_atom_vocab
from collections import deque

add = lambda x,y : x + y if type(x) is int else (x[0] + y, x[1] + y)

class MolGraph(object):

    BOND_LIST = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 
    MAX_POS = 40

    def __init__(self, smiles, init_atoms, root_atoms=None, shuffle_roots=True):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        self.mol_graph = self.build_mol_graph()
        self.init_atoms = set(init_atoms)
        self.root_atoms = self.get_root_atoms() if root_atoms is None else root_atoms
        if len(self.root_atoms) > 0:
            if shuffle_roots: random.shuffle(self.root_atoms)
            self.order = self.get_bfs_order()

    def debug(self):
        for atom in self.mol.GetAtoms():
            if atom.GetIdx() in self.init_atoms:
                atom.SetAtomMapNum(atom.GetIdx())
        print( Chem.MolToSmiles(self.mol) )
        print('root', self.root_atoms)
        print('init', self.init_atoms)
        for x in self.order:
            print(x)

    def get_root_atoms(self):
        roots = []
        for idx in self.init_atoms:
            atom = self.mol.GetAtomWithIdx(idx)
            bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in self.init_atoms]
            if len(bad_neis) > 0:
                roots.append(idx)
        return roots

    def build_mol_graph(self):
        mol = self.mol
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        for atom in mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = MolGraph.BOND_LIST.index( bond.GetBondType() )
            graph[a1][a2]['label'] = btype
            graph[a2][a1]['label'] = btype

        return graph

    def get_bfs_order(self):
        order = []
        visited = set(self.init_atoms)
        queue = deque( [self.mol.GetAtomWithIdx(k) for k in self.root_atoms] )

        for a in self.init_atoms:
            self.mol_graph.nodes[a]['pos'] = 0

        for i,root in enumerate(self.root_atoms):
            self.mol_graph.nodes[root]['pos'] = i + 1  # interior atoms are 0, boundary atoms are 1,2,...

        pos_id = len(self.root_atoms)
        while len(queue) > 0:
            x = queue.popleft()
            x_idx = x.GetIdx()
            for y in x.GetNeighbors():
                y_idx = y.GetIdx()
                if y_idx in visited: continue

                frontier = [x_idx] + [a.GetIdx() for a in list(queue)]
                bonds = [0] * len(frontier)
                y_neis = set([z.GetIdx() for z in y.GetNeighbors()])

                for i,z_idx in enumerate(frontier):
                    if z_idx in y_neis: 
                        bonds[i] = self.mol_graph[y_idx][z_idx]['label']

                order.append( (x_idx, y_idx, frontier, bonds) )
                pos_id += 1
                self.mol_graph.nodes[y_idx]['pos'] = min(MolGraph.MAX_POS - 1, pos_id)
                visited.add( y_idx )
                queue.append(y)

            order.append( (x_idx, None, None, None) )

        return order
    
    @staticmethod
    def tensorize(mol_batch, avocab=common_atom_vocab):
        graph_tensors, graph_batchG = MolGraph.tensorize_graph([x.mol_graph for x in mol_batch], avocab)
        graph_scope = graph_tensors[-1]

        add = lambda a,b : None if a is None else a + b
        add_list = lambda alist,b : None if alist is None else [a + b for a in alist]

        all_orders = []
        all_init_atoms = []
        for i,hmol in enumerate(mol_batch):
            offset = graph_scope[i][0]
            order = [(x + offset, add(y, offset), add_list(z, offset), t) for x,y,z,t in hmol.order]
            init_atoms = [x + offset for x in hmol.init_atoms]
            all_orders.append(order)
            all_init_atoms.append(init_atoms)

        return graph_batchG, graph_tensors, all_init_atoms, all_orders

    @staticmethod
    def tensorize_graph(graph_batch, vocab):
        fnode,fmess = [None],[(0,0,0)] 
        agraph,bgraph = [[]], [[]] 
        scope = []
        edge_dict = {}
        all_G = []

        for bid,G in enumerate(graph_batch):
            offset = len(fnode)
            scope.append( (offset, len(G)) )
            G = nx.convert_node_labels_to_integers(G, first_label=offset)
            all_G.append(G)
            fnode.extend( [None for v in G.nodes] )

            for v, attr in G.nodes(data='label'):
                G.nodes[v]['batch_id'] = bid
                fnode[v] = (vocab[attr], G.nodes[v]['pos'])
                agraph.append([])

            for u, v, attr in G.edges(data='label'):
                fmess.append( (u, v, attr) )
                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid
                agraph[v].append(eid)
                bgraph.append([])

            for u, v in G.edges:
                eid = edge_dict[(u, v)]
                for w in G.predecessors(u):
                    if w == v: continue
                    bgraph[eid].append( edge_dict[(w, u)] )

        fnode[0] = fnode[1]
        fnode = torch.LongTensor(fnode)
        fmess = torch.LongTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)
        return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)

