


data_mount="bo.nov29/datasets/squad/3"
dir_data="/data"

dir_output="/output"


embed_matrix_file="embed_matrix.npy"
pass_max_len=400
ques_max_len=30
embed_size=50
num_units=64
clip_norm=5.0
optimizer="adam"
lr=2e-2
n_epoch=1
batch_size=32
keep_prob=1.0


if [ "$1" = "gpu" ]
then
echo "Using floyd $1"
floyd run --gpu --env tensorflow-1.4:py2 --data $data_mount:/data "python console.py $dir_data $dir_output $embed_matrix_file $pass_max_len $ques_max_len $embed_size $num_units $clip_norm $optimizer $lr $n_epoch $batch_size $keep_prob"
elif [ "$1" = "cpu" ]
then
echo "Using floyd $1"
floyd run --env tensorflow-1.4:py2 --data $data_mount:/data "python console.py $dir_data $dir_output $embed_matrix_file $pass_max_len $ques_max_len $embed_size $num_units $clip_norm $optimizer $lr $n_epoch $batch_size $keep_prob"
else
echo "Please indicate cpu or gpu"
fi
