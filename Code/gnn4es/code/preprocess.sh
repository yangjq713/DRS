prepare_train_data(){
    local dataset=$1
	local phase=$2
    local fold=$3
    local topk=$4
    local model_name=$5
	python prepare_train_data.py  --dataset "$dataset" --phase "$phase" --fold "$fold" --topk "$topk" --model_name "$model_name"
}

prepare_test_data(){
    local dataset=$1
	local phase=$2
    local fold=$3
    local topk=$4
    local model_name=$5
	python prepare_test_data.py  --dataset "$dataset" --phase "$phase" --fold "$fold" --topk "$topk" --model_name "$model_name"
}

rm_data_all(){
    for dataset in "dbpedia" "lmdb" "faces"
    do
        for fold in 0 1 2 3 4 
        do
	        python rm_gene_file.py  --dataset "$dataset" --fold "$fold"
        done
    done
    echo "rm gene file success"
}

rm_data_one(){
    local dataset=$1
    for fold in 0 1 2 3 4 
    do
        python rm_gene_file.py  --dataset "$dataset" --fold "$fold"
    done
    echo "rm gene file success"
}

process_all(){
    rm_data_all
    local model_name=$1
    for dataset in "dbpedia" "lmdb" "faces"
    do
        for topk in 5 10
        do
            for fold in 0 1 2 3 4
            do
                for phase in "train" "valid"
                do
                    para="{
                            \"dataset\":\"$dataset\",
                            \"topk\":\"$topk\",
                            \"fold\":\"$fold\",
                            \"phase\":\"$phase\"
                            \"model_name\":\"$model_name\"
                        }"
                    echo $para
                    prepare_train_data "$dataset" "$phase" "$fold" "$topk" "$model_name"
                done
                echo "construct test data"
                prepare_test_data "$dataset" "test" "$fold" "$topk" "$model_name"
            done
        done
    done
}

process_one_dataset(){
    local dataset=$1
    local model_name=$2
    # rm_data_one $dataset
    for topk in 5 10
    do
        for fold in 0 1 2 3 4
        do
            for phase in "train" "valid"
            do
                para="{
                        \"dataset\":\"$dataset\",
                        \"topk\":\"$topk\",
                        \"fold\":\"$fold\",
                        \"phase\":\"$phase\",
                        \"model_name\":\"$model_name\"
                    }"
                echo $para
                prepare_train_data "$dataset" "$phase" "$fold" "$topk" "$model_name"
            done
            echo "construct test data"
            prepare_test_data "$dataset" "test" "$fold" "$topk" "$model_name"
        done
    done
}

if [ "$1" = "all" ]; then
    process_all "$2"
elif [ "$1" = "one" ]; then
    process_one_dataset "$2" "$3"
fi
