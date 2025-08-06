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
    
    
    