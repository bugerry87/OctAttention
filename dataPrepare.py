'''
Author: fuchy@stu.pku.edu.cn
Description: Prepare data for traning and testing.
             *.mat is generated by dataPrepare from *.ply
             *.mat data structure cell{N*4} :
                N: point clouds number; N=1 in '*.mat'
                    {
                        [TreePoints*K*C] Octree data sequence generated from PQ (Quantized point cloud): N*K*C [n,7,6] array
                                         N[n treepoints]  K[7 ancestors]   C[oct code,level,octant,position(xyz)]
                4:      {Location} Original geometric coordinate P (n*3)
                        {qs,offset,Lmax,name} side information; Quantized point cloud PQ = (P-offset)/qs; The depth of PQ; The name of P (point cloud)
                    }
All rights reserved.
'''
import glob
from os import path
from Preparedata.data import dataPrepare
from networkTool import CPrintl
from argparse import ArgumentParser


def init_main_args(parents=[]):
    main_args = ArgumentParser(
        description="Data Prepare",
        conflict_handler='resolve',
        parents=parents
    )
    
    main_args.add_argument(
        '--input', '-i',
        help='Path to dataset directory'
    )
     
    main_args.add_argument(
        '--output', '-o',
        help='Path for converted data'
    )
    
    main_args.add_argument(
        '--prefix', '-p',
        default='MPEG_',
        help='Prefix for filename'
    )

    main_args.add_argument(
        '--qlevel', '-q',
        type=int,
        default=12,
        help='Quantization Level'
    )

    main_args.add_argument(
        '--offset', '-O',
        default='min',
        help='Offset method'
    )
    
    main_args.add_argument(
        '--log',
        default='./Preparedata/dataPrepare.log',
        help='Path for log file'
    )
    return main_args

if __name__=="__main__":
    args = init_main_args().parse_args()

    ptNamePrefix = args.prefix
    qlevel = 2/(2**args.qlevel-1)
    printl = CPrintl(args.log)
    for n, file in enumerate(glob.glob(args.input)):
        dataPrepare(
            file,
            saveMatDir=args.output,
            ptNamePrefix=f'{args.prefix}{n:06d}_',
            offset=args.offset,
            qs=qlevel,
            normalize=True,
            rotation=False,
        )
        printl(file)