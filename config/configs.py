class Config_Training(object):
    DIR_TRAIN = ""
    DIR_VAL = ""
    DIR_TEST = ""
    NUM_EPOCHS = 100
    LR = 0.0001
    n_classes = 2
    device = "cuda:0"
    pretrained = None
    lambda_loss = 1
    lambda_loss_segment = 0.01
    lambda_loss_a = 0.1
    lambda_loss_b = 0.001
    lambda_loss_c = 0.01
    batch_size = 32
    num_workers = 2
    print_freq = 10
    fiducial_point_gaps = [1, 2, 3, 5, 6, 10, 15, 30] 
    fiducial_point_num = [31, 16, 11, 7, 6, 4, 3, 2]
    col_gap = 0
    row_gap = 0
    tps = True
    fiducial_num = fiducial_point_num[col_gap],fiducial_point_num[row_gap]
    
    
    
    
    
