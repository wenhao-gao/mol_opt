fpath = 'model-300-3-direct/test_withReagents.cbond.num_cands_core{}.txt'
for n in [6, 8, 10, 12, 14, 16, 18, 20]:
    with open(fpath.format(n), 'r') as fid:
        tot = 0.
        cands = 0.
        for line in fid:
            tot += 1
            cands += int(line.strip())
    print('Average num cands using n = {}, {}'.format(n, cands/tot))



