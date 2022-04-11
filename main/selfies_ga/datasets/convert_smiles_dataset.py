#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 15:23:30 2019

@author: akshat
"""
from selfies import encoder, decoder


fname = './smiles_qm9.txt'
with open(fname) as f:
    content = f.readlines()

content = [x.strip() for x in content] 
content  = [x.split(',')[1] for x in content]
content = content[1:]


for item in content:
    selfie_str = encoder(item)
    write_fname = './SELFIES_qm9.txt'
    with open(write_fname, "a") as myfile:
        myfile.write(selfie_str + "\n")