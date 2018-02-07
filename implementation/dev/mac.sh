
dir_data="./$1/data"
dir_output="./$1/output"

if [ "$1" = "mac0" ]
then
echo "$1"
pass_max_length=13
ques_max_length=7
batch_size=97
embed_size=50
elif [ "$1" = "mac1" ]
then
echo "$1"
pass_max_length=29
ques_max_length=17
batch_size=13
embed_size=50
elif [ "$1" = "mac2" ]
then
echo "$1"
pass_max_length=97
ques_max_length=29
batch_size=17
embed_size=50
elif [ "$1" = "mac3" ]
then
echo "$1"
pass_max_length=766
ques_max_length=60
batch_size=17
embed_size=50
elif [ "$1" = "mac4" ]
then
echo "$1"
pass_max_length=199
ques_max_length=37
batch_size=17
embed_size=50
else
echo "Please indicate version of dataset to use"
fi


num_units=10

dropout=0.5
do_clip=True
clip_norm=5
optimizer="adam"
lr=10
n_epoch=1
train_batches_sub_path="data_feed_ready/train.batches"
valid_json_sub_path="data_json/valid.json"
valid_passage_tokens_sub_path="data_token/valid.passage"
valid_question_ids_sub_path="data_token/valid.question_id"
valid_batches_sub_path="data_feed_ready/valid.batches"
test_json_sub_path="data_json/test.json"
test_passage_tokens_sub_path="data_token/test.passage"
test_question_ids_sub_path="data_token/test.question_id"
test_batches_sub_path="data_feed_ready/test.batches"

python ../code/console.py $dir_data $dir_output $pass_max_length $ques_max_length $batch_size $embed_size $num_units $dropout $do_clip $clip_norm $optimizer $lr $n_epoch $train_batches_sub_path $valid_json_sub_path $valid_passage_tokens_sub_path $valid_question_ids_sub_path $valid_batches_sub_path $test_json_sub_path $test_passage_tokens_sub_path $test_question_ids_sub_path $test_batches_sub_path
