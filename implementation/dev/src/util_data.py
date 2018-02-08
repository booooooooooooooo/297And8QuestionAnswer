


'''
make mask matrix

pad/truncate train token_ids, valid token_ids and test token_ids

adjust answers for truncated passages

if a_e >= pass_max_length:
    a_e = pass_max_length - 1
if a_s >= pass_max_length:
    a_s = random.randrange(0, pass_max_length)

'''
#TODO
class DataUtil:
    def pad_train(self, )
