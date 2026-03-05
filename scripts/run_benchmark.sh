export KIMINA7="AI-MO/Kimina-Autoformalizer-7B"
export GOEDEL32="Goedel-LM/Goedel-Formalizer-V2-32B"
export GOEDEL8="Goedel-LM/Goedel-Formalizer-V2-8B"
export GPT120="openai/gpt-oss-120b"
export GPT20="openai/gpt-oss-20b"
export ATLAS_D="XiaoyangLiu-sjtu/ATLAS_Translator_D"
export ATLAS_Q="XiaoyangLiu-sjtu/ATLAS_Translator_Q"
export ATLAS_L="XiaoyangLiu-sjtu/ATLAS_Translator_L"
export QWEN35_27="Qwen/Qwen3.5-27B"
export QWEN35_9="Qwen/Qwen3.5-9B"
export QWEN35_4="Qwen/Qwen3.5-4B"
export HERALD="FrenzyMath/Herald_translator"

export SPLIT="definition"

for MODEL in $GOEDEL32 $GOEDEL8 $GPT120 $GPT20 $ATLAS_Q $HERALD $KIMINA7 ; do
	echo "Running $SPLIT with model $MODEL"
	export OUTPUT_PATH=outputs/$(echo "${MODEL,,}.${SPLIT}.json" | sed 's/\//./g');
	python -m benchmarks.run_benchmark \
		--model=$MODEL\
		--dataset="offendo/math-atlas" \
		--output $OUTPUT_PATH \
		--max-tokens=10000 \
		--temperature=0.2 \
		--top-p=0.95 \
		--seed=1234 \
		--item-type=$SPLIT;
	echo "...done!"

done
