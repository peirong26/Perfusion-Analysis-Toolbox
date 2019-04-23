import argparse

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def parse_config():

    parser = argparse.ArgumentParser(description = "CTP Colormaps Calculation")
    
    parser.add_argument('--image_type', type = str, default = 'CTP', help = 'Image type: CTP/MRP')

    ################## Constants Settings ##################
    parser.add_argument('--k_ct', type = float, default = 1.0, help = 'Constant k_ct (g/ml/HU) for CTP')

    parser.add_argument('--k_mr', type = float, default = 1.0, help = 'Constant k_mr for MRP')
    parser.add_argument('--TE', type = float, default = 0.025, help = 'Constant TE (ms) for MRP')
    parser.add_argument('--TR', type = float, default = 1.55, help = 'Constant TR (s) for MRP')
    # Usually, need filter for MRP, no need for CTP
    parser.add_argument('--use_filter', type = bool, default = False, help = 'Whether use low-pass filtering for CTC')
    parser.add_argument('--mrp_s0_threshold', type = float, default = 0.05, help = 'Threshold for finding MRP bolus arrival time ')
    parser.add_argument('--ctp_s0_threshold', type = float, default = 0.05, help = 'Threshold for finding CTP bolus arrival time ')

    parser.add_argument('--to_tensor', type = bool, default = True, help = 'Whether need to convert to torch.tensor')
    parser.add_argument('--mask', type = list, default = [[], [0,489], [60,501]], help = "Used as BackGround Code for MRP, \
        while BrainMask -300 for CTP (UNC)") 
        # default: 0 for MRP, [[], [0,489], [60,501]] for CTP (UNC)

    args = parser.parse_args()

    return args

