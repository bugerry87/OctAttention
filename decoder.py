'''
Author: fuchy@stu.pku.edu.cn
Date: 2021-09-17 23:30:48
LastEditTime: 2021-12-02 22:18:56
LastEditors: FCY
Description: decoder
FilePath: /compression/decoder.py
All rights reserved.
'''

from argparse import ArgumentParser
import os, time, glob, pt, torch, numpyAc
import numpy as np
from tqdm import tqdm
from Octree import DeOctree, dec2bin
from dataset import default_loader as matloader
from collections import deque
#from networkTool import *
from networkTool import device, levelNumK, bptt, reload
from encoderTool import bpttRepeatTime, generate_square_subsequent_mask
from octAttention import model
batch_size = 1 

'''
description: decode bin file to occupancy code
param {str;input bin file name} binfile
param {N*1 array; occupancy code, only used for check} oct_data_seq
param {model} model
param {int; Context window length} bptt
return {N*1,float}occupancy code,time
'''
def decodeOct(binfile,oct_data_seq,model,bptt):
    model.eval()
    with torch.no_grad():
        elapsed = time.time()

        KfatherNode = [[255,0,0]] * levelNumK
        nodeQ = deque()
        oct_seq = []
        src_mask = generate_square_subsequent_mask(bptt).to(device)

        input = torch.zeros((bptt, batch_size, levelNumK, 3)).long().to(device)
        padinginbptt = torch.zeros((bptt, batch_size, levelNumK, 3)).long().to(device)
        bpttMovSize = bptt//bpttRepeatTime
        # input torch.Size([256, 32, 4, 3]) bptt,batch_sz,kparent,[oct,level,octant]
        # all of [oct,level,octant] default is zero

        output = model(input,src_mask,[])
        freqsinit = torch.softmax(output[-1],1).squeeze().cpu().detach().numpy()
        oct_len = len(oct_data_seq)
        dec = numpyAc.arithmeticDeCoding(None,oct_len,255,binfile)
        root =  decodeNode(freqsinit,dec)
        nodeId = 0
        
        # for old dataset (e.g. from OctTrans)
        # KfatherNode = KfatherNode[2:]+[[root,1,1]] #+ [[root,1,1]] # for padding for first row # (old data)
        # for new dataset (e.g. from dataPrepare.py)
        KfatherNode = KfatherNode[3:]+[[root,1,1]] + [[root,1,1]] # for padding for first row # ( the parent of root node is root itself)
        
        nodeQ.append(KfatherNode) 
        oct_seq.append(root) #decode the root  
        
        with tqdm(total=  oct_len+10) as pbar:
            while True:
                father = nodeQ.popleft()
                childOcu = dec2bin(father[-1][0])
                childOcu.reverse()
                faterLevel = father[-1][1] 
                for i in range(8):
                    if(childOcu[i]):
                        faterFeat = [[father+[[root,faterLevel+1,i+1]]]] # Fill in the information of the node currently decoded [xi-1, xi level, xi octant]
                        faterFeatTensor = torch.Tensor(faterFeat).long().to(device)
                        faterFeatTensor[:,:,:,0] -= 1

                        # shift bptt window
                        offsetInbpttt = (nodeId)%(bpttMovSize) # the offset of current node in the bppt window
                        if offsetInbpttt==0: # a new bptt window
                            input = torch.vstack((input[bpttMovSize:],faterFeatTensor,padinginbptt[0:bpttMovSize-1]))
                        else:
                            input[bptt-bpttMovSize+offsetInbpttt] = faterFeatTensor

                        output = model(input,src_mask,[])
                        
                        Pro = torch.softmax(output[offsetInbpttt+bptt-bpttMovSize],1).squeeze().cpu().detach().numpy()

                        root =  decodeNode(Pro,dec)
                        nodeId += 1
                        pbar.update(1)
                        KfatherNode = father[1:]+[[root,faterLevel+1,i+1]]
                        nodeQ.append(KfatherNode)
                        if(root==256 or nodeId==oct_len):
                            assert len(oct_data_seq) == nodeId # for check oct num
                            Code = oct_seq
                            return Code,time.time() - elapsed
                        oct_seq.append(root)
                    assert oct_data_seq[nodeId] == root,'please check KfaterNode in line 60' # for check

def decodeNode(pro,dec):
    root = dec.decode(np.expand_dims(pro,0))
    return root+1

def init_main_args(parents=[]):
    """
    Initialize an ArgumentParser for this module.
    
    Args:
        parents: A list of ArgumentParsers of other scripts, if there are any.
        
    Returns:
        main_args: The ArgumentParsers.
    """
    main_args = ArgumentParser(
        description="OctAttention Decoder",
        conflict_handler='resolve',
        parents=parents
        )
    
    main_args.add_argument(
        '--samples', '-X',
        metavar='WILDCARD',
        required=True,
        nargs='+',
        help='A wildcard to the point cloud files'
        )
     
    main_args.add_argument(
        '--features', '-F',
        metavar='PATH',
        required=True,
        help='Path to pre-processed features'
        )
    
    main_args.add_argument(
        '--output', '-Y',
        metavar='Path',
        required=True,
        help='Path for restored point clouds'
        )
    
    main_args.add_argument(
        '--model', '-M',
        metavar='PATH',
        default='modelsave/lidar/encoder_epoch_00801460.pth',
        help='Path to model'
        )
    return main_args


if __name__=="__main__":
    args = init_main_args().parse_known_args()
    model = model.to(device)
    saveDic = reload(
        None,
        args.model,
        multiGPU=False
    )
    model.load_state_dict(saveDic['encoder'])

    for oriFile in glob.glob(args.samples):
        binfile = oriFile
        ptName = os.path.splitext(os.path.basename(oriFile))[0]
        matName = os.path.join(args.features, ptName + '.mat')
        cell, mat = matloader(matName)

        # Read Sideinfo
        oct_data_seq = np.transpose(mat[cell[0,0]]).astype(int)[:,-1:,0]# for check
        
        p = np.transpose(mat[cell[1,0]]['Location']) # ori point cloud
        offset = np.transpose(mat[cell[2,0]]['offset'])
        qs = mat[cell[2,0]]['qs'][0]

        Code, elapsed = decodeOct(binfile, oct_data_seq, model, bptt)
        print('decode succee,time:', elapsed)
        print('oct len:', len(Code))

        # DeOctree
        ptrec = DeOctree(Code)
        # Dequantization
        DQpt = (ptrec*qs+offset)
        outp = os.path.join(args.output, ptName + '.ply')
        pt.write_ply_data(outp, DQpt)
        pt.pcerror(outp, DQpt, None, '-r 1', None).wait()