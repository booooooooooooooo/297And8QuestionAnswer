'''
All hyperparameters of model
'''
class Config:
    def __init__(self, batch_s, embed_s, num_units, pass_l, n_epoch, lr):
        self.batch_s = batch_s
        self.embed_s = embed_s
        self.num_units = num_units
        self.pass_l = pass_l
        self.n_epoch = n_epoch
        self.lr = lr
