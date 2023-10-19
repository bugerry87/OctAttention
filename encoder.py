'''
Author: fuchy@stu.pku.edu.cn
Description: this file encodes point cloud
FilePath: /compression/encoder.py
All rights reserved.
'''
from argparse import ArgumentParser
from Preparedata.data import dataPrepare
from encoderTool import main
from networkTool import reload, device
from octAttention import model
import glob, datetime, os
import pt as pointCloud

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
        nargs='+',
        help='A wildcard to the point cloud files'
        )
     
    main_args.add_argument(
        '--features', '-F',
        metavar='PATH',
        required=True,
        help='path to pre-processed features'
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
    args = init_main_args().parse_known_args()
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
        ptName = os.path.splitext(os.path.basename(oriFile))[0] 
        for qlevel in [12]:
            matFile, DQpt, normalizePt = dataPrepare(
                oriFile,
                saveMatDir=args.featues,
                offset='min',
                qs=2/(2**qlevel-1),
                rotation=False,
                normalize=True
            )
            main(matFile, model, actualcode=True, printl=print) # actualcode=False: bin file will not be generated
            print('_'*50,'pc_error','_'*50)
            pointCloud.pcerror(
                normalizePt,
                DQpt,
                None,
                '-r 1',
                None
            ).wait()
            print('cd %e'%pointCloud.distChamfer(normalizePt, DQpt))
