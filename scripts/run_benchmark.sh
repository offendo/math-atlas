export KIMINA7="AI-MO/Kimina-Autoformalizer-7B"
export GOEDEL32="Goedel-LM/Goedel-Formalizer-V2-32B"
export GOEDEL8="Goedel-LM/Goedel-Formalizer-V2-8B"
export GPT120="openai/gpt-oss-120b"
export GPT20="openai/gpt-oss-20b"
export ATLAS_D="XiaoyangLiu-sjtu/ATLAS_Translator_D"
export ATLAS_Q="XiaoyangLiu-sjtu/ATLAS_Translator_Q"
export ATLAS_L="XiaoyangLiu-sjtu/ATLAS_Translator_L"
export QWEN3_30="Qwen/Qwen3-30B-A3B-Thinking-2507"
export QWEN3_8="Qwen/Qwen3-8B"
export QWEN3_4="Qwen/Qwen3-4B-Thinking-2507"
export HERALD="FrenzyMath/Herald_translator"

export SPLITS="theorem exercise example"
export SPLIT_NAME="statements"
# export SPLITS="definition"
# export SPLIT_NAME="definitions"

export CUDA_VISIBLE_DEVICES=2

for MODEL in $ATLAS_L $GOEDEL8; do
	echo "Running $SPLITS with model $MODEL"
	export OUTPUT_PATH=outputs/$(echo "${MODEL,,}.${SPLIT_NAME}.json" | sed 's/\//./g');
	python -m benchmarks.run_benchmark \
		 $SPLITS \
		--model=$MODEL\
		--dataset="offendo/math-atlas" \
		--output $OUTPUT_PATH \
		--data-parallel-size=1 \
		--max-tokens=10000 \
		--temperature=0.2 \
		--top-p=0.95 \
		--seed=1234
	echo "...done!"
done
