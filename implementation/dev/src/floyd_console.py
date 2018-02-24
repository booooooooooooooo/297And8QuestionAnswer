
from train import train

if __name__ == "__main__":

    dir_data="/data"
    dir_output="/output"


    embed_size = 100
    pass_max_len = 400
    ques_max_length = 30
    num_units = 64
    clip_norm = 10
    lr = 2e-3
    n_epoch = 1
    reg_scale = 0.001
    batch_size = 32
    sample_size = 200

    train(dir_data, embed_size, pass_max_len, ques_max_length, num_units, clip_norm, lr, n_epoch, reg_scale, batch_size, sample_size, dir_output)
