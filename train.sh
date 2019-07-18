model_dir=../model-LSTM

if [ ! -e $model_dir ] ; then
	echo 'create',$model_dir
    mkdir -p $model_dir
elif [ ! -d $model_dir ] ; then
    echo "$dir already exists but is not a directory" 1>&2
else 
	echo $model_dir 'exists'
fi

output_dir=./prediction

if [ ! -e $output_dir ] ; then
	echo 'create',$output_dir
    mkdir -p $output_dir
elif [ ! -d $output_dir ] ; then
    echo "$dir already exists but is not a directory" 1>&2
else 
	echo $output_dir 'exists'
fi

CUDA_VISIBLE_DEVICES=0 python3 run.py \
--run_mode train \
--model_path $model_dir/model-36000.pt \
--model_name GRNN_LSTM_1 \
--save_steps 1000 \
--ner_class_config ../nyt-data/nyt.ner.label \
--re_class_config ../nyt-data/nyt.re.label \
--shuffle_file ../nyt-data/$0.shuffle \
--train_file ../nyt-data/nyt.train.pos.json \
--test_file ../nyt-data/nyt.test.json \
--data_set NYT \
--task TE \
--use_bert 1 \
--feature_based 0 \
--use_word_embeding 0 \
--bert_dir /home/zack/bert-base-uncased \
--word_embeding_path /home/zack/relation-new/nyt-data/vocab/dataset.h5 \
--vocab_path /home/zack/relation-new/nyt-data/vocab/dataset.voc \
--build_vocab /home/zack/relation-new/nyt-data/vocab/dataset.build.voc \
--use_char_embed 1 \
--charcab_path /home/zack/relation-new/nyt-data/vocab/dataset.char \
--bert_dir /home/zack/bert-base-uncased \
--save_model_dir $model_dir \
--test_log ../test.log \
--lr 1e-5 \
--max_sentence_length 512 \
--epoch 40 \
--edge_file /home/zack/relation/data/edge_type_lexicon \
--infer_entity 1 \
--max_negative_re 100 \
--max_dist 50 \
--batch_size 1 \
--relation_entity_span 1 \
--consider_none_relaton 0 \
--pos_vocab /home/zack/relation/data/semeval.pos.pkl \
--show_attention 0 \
--deepest 6 