dir_data="../data/data_clean"
dir_output="../output"

embed_matrix_file="embed_matrix.npy"
pass_max_len=99
ques_max_len=47
embed_size=50
num_units=50
clip_norm=5.0
optimizer="adam"
lr=1.0
n_epoch=1
batch_size=77
keep_prob=0.5



python console.py $dir_data $dir_output $embed_matrix_file $pass_max_len $ques_max_len $embed_size $num_units $clip_norm $optimizer $lr $n_epoch $batch_size $keep_prob
