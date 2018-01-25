'''
TO run on floyd:

data_mount="bo.nov29/datasets/squad/2"
datafolder = "/data"
outputfolder="/output"
pass_max_length=766
ques_max_length=60
batch_size=97
embed_size=50

floyd run --gpu --env tensorflow-1.4:py2 --data $data_mount:/data "python console.py ............"



To run on mac:


datafolder = "../mac/data"
outputfolder="../mac/output"
pass_max_length=13
ques_max_length=7
batch_size=97
embed_size=50

python console.py ..........

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Running match_lstm_ans_ptr')
    parser.add_argument('dir_data')
    parser.add_argument('dir_output')

    parser.add_argument('pass_max_length')
    parser.add_argument('ques_max_length')
    parser.add_argument('batch_size')
    parser.add_argument('embed_size')
    parser.add_argument('num_units')
    parser.add_argument('dropout')
    parser.add_argument('do_clip')
    parser.add_argument('clip_norm')
    parser.add_argument('optimizer')
    parser.add_argument('lr')
    parser.add_argument('n_epoch')
    parser.add_argument('train_batches_sub_path')

    parser.add_argument('valid_json_sub_path')
    parser.add_argument('valid_passage_tokens_sub_path')
    parser.add_argument('valid_question_ids_sub_path')
    parser.add_argument('valid_batches_sub_path')

    parser.add_argument('test_json_sub_path')
    parser.add_argument('test_passage_tokens_sub_path')
    parser.add_argument('test_question_ids_sub_path')
    parser.add_argument('test_batches_sub_path')

    args = parser.parse_args()


    saved_model_list = train(int (args.pass_max_length), int (args.ques_max_length), int (args.batch_size), int (args.embed_size), int (args.num_units), float(args.dropout), bool(args.do_clip), float(args.clip_norm), args.optimizer, float (args.lr), int(args.n_epoch), args.train_batches_sub_path, args.dir_data, args.dir_output)

    best_graph_info = valid(args.json_file, args.passage_tokens_file, args.question_ids_file, args.batches_file, args.graph_path_list_file )

    test_score = test(args.json_file, args.passage_tokens_file, args.question_ids_file, args.batches_file, args.best_graph_info_file)

#not related to preprocess



######train############################################
#######################################################
num_units=10
dropout=0.5
do_clip=false
clip_norm=5
optimizer="adam"
lr=10
n_epoch=5
batches_file="/data/data_feed_ready/train.batches"
dir_to_save_graph="/output/tf_graph/"
floyd run --env tensorflow-1.4:py2 --data $datafolder:/data "python train.py $pass_max_length $ques_max_length $batch_size $embed_size $num_units $dropout $do_clip $clip_norm $optimizer $lr $n_epoch $batches_file $dir_to_save_graph"



######valid############################################
#######################################################
trainoutputfolder="TBD"
valid_json="/data/data_json/valid.json"
valid_passage="/data/data_token/valid.passage"
valid_question_id="/data/data_token/valid.question_id"
valid_batches="/data/data_feed_ready/valid.batches"
dir_to_save_graph="/model/tf_graph/"
best_graph_info_json_file="/output/best_graph_info.json"
floyd run --env tensorflow-1.4:py2 --data $datafolder:/data --data $trainoutputfolder:/model "python valid.py $valid_json $valid_passage $valid_question_id $valid_batches $dir_to_save_graph $best_graph_info_json_file"





######test############################################
######################################################
validoutputfolder="TBD"
test_json="/data/data_json/test.json"
test_passage="/data/data_token/test.passage"
test_question_id="/data/data_token/test.question_id"
test_batches="/data/data_feed_ready/test.batches"
best_graph_info_json_file="/model/best_graph_info.json"
test_score_info_file="/output/test_score_info"
floyd run --env tensorflow-1.4:py2 --data $datafolder:/data --data $validoutputfolder:/model "python test.py $test_json $test_passage $test_question_id $test_batches $best_graph_info_json_file $test_score_info_file"
