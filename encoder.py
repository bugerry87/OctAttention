'''
Author: fuchy@stu.pku.edu.cn
Description: this file encodes point cloud
FilePath: /compression/encoder.py
All rights reserved.
'''
from argparse import ArgumentParser
from Preparedata.data import dataPrepare
from encoderTool import compress
from dataset import default_loader as matloader
from networkTool import reload, device, levelNumK
from octAttention import model
import glob, datetime, os
import numpy as np

############## warning ###############
## decoder.py and test.py rely on this model here
## do not move this lines to somewhere else


###########LiDar##############
#GPCC_MULTIPLE = 2**20

def init_main_args(parents=[]):
    """
    Initialize an ArgumentParser for this module.
    
    Args:
        parents: A list of ArgumentParsers of other scripts, if there are any.
        
    Returns:
        main_args: The ArgumentParsers.
    """
    main_args = ArgumentParser(
        description="OctAttention Encoder",
        conflict_handler='resolve',
        parents=parents
        )
    
    main_args.add_argument(
        '--samples', '-X',
        metavar='WILDCARD',
        required=True,
        help='A wildcard to the point cloud files'
        )
     
    main_args.add_argument(
        '--features', '-F',
        metavar='PATH',
        help='Path to pre-processed features'
        )
    
    main_args.add_argument(
        '--output', '-Y',
        metavar='PATH',
        required=True,
        help='Path for compressed output'
        )
    
    main_args.add_argument(
        '--model', '-M',
        metavar='PATH',
        default='modelsave/lidar/encoder_epoch_00801460.pth',
        help='Path to model'
        )
    
    main_args.add_argument(
        '--quantization', '-q',
        metavar='INT',
        type=int,
        default=12,
        )
    return main_args

if __name__=="__main__":
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    args = init_main_args().parse_known_args()[0]
    model = model.to(device)
    saveDic = reload(
        None,
        args.model,
        multiGPU=False
    )
    model.load_state_dict(saveDic['encoder'])

    print('_'*50, 'OctAttention V0.4', '_'*50)
    print(datetime.datetime.now().strftime('%Y-%m-%d:%H:%M:%S'))
    print('load checkpoint', saveDic['path'])

    for oriFile in glob.glob(args.samples):
        print(oriFile)
        for qlevel in [12]:
            if oriFile.endswith('.mat'):
                matFile = oriFile
            else:
                matFile, DQpt, normalizePt = dataPrepare(
                    oriFile,
                    saveMatDir=args.featues,
                    offset='min',
                    qs=2/(2**qlevel-1),
                    rotation=False,
                    normalize=True
                )
                pass

            #main(matFile, model, actualcode=True, printl=print) # actualcode=False: bin file will not be generated
            cell, mat = matloader(matFile)
            oct_data_seq = np.transpose(mat[cell[0,0]]).astype(int)[:,-levelNumK:,0:6] 
            p = np.transpose(mat[cell[1,0]]['Location'])
            ptNum = p.shape[0]
            ptName = os.path.splitext(os.path.basename(matFile))[0]
            output = os.path.join(args.output, ptName + ".bin")
            binsz, oct_len, elapsed, binszList, octNumList = compress(
                oct_data_seq,
                output,
                model,
                True,
                print,
                False
            )

            print("ptName: ", ptName)
            print("time(s):", elapsed)
            print("ori file", matFile)
            print("ptNum:", ptNum)
            print("binsize(b):", binsz)
            print("bpip:", binsz/ptNum)
            print("pre sz(b) from Q8:",(binszList))
            print("pre bit per oct from Q8:", (binszList / octNumList))
            print('octNum:', octNumList)
            print("bit per oct:", binsz/oct_len)
            print("oct len:", oct_len)
            pass
        pass
    pass
