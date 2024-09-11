import os
import sys

def set_envs():
    #Conifer folder
    sys.path.append("/afs/cern.ch/work/p/pviscone/conifer/conifer")

    os.environ["PATH"] = "/home/Xilinx/Vivado/2023.1/bin:/home/Xilinx/Vitis_HLS/2023.1/bin:" + os.environ["PATH"]
    os.environ["XILINX_AP_INCLUDE"] = "/afs/cern.ch/work/p/pviscone/conifer/XilinxInclude"
    os.environ["JSON_ROOT"] = "/afs/cern.ch/work/p/pviscone/conifer"
