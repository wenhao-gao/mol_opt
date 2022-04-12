from main.molpal.molpal.pools.base import MoleculePool, EagerMoleculePool
from main.molpal.molpal.pools.lazypool import LazyMoleculePool

def pool(pool: str, *args, **kwargs):
    try:
        return {
            'eager': MoleculePool,
            'lazy': LazyMoleculePool
        }[pool](*args, **kwargs)
    except KeyError:
        print(f'WARNING: Unrecognized pool type: "{pool}". Defaulting to EagerMoleculePool.')
        return MoleculePool(*args, **kwargs)