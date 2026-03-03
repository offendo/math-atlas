# export MODEL="AI-MO/Kimina-Autoformalizer-7B"
export MODEL="Goedel-LM/Goedel-Formalizer-V2-32B"
export SPLIT="theorem"
export OUTPUT_PATH=$(echo "${MODEL,,}.${SPLIT}.json" | sed 's/\//./g')

python -m benchmarks.run_benchmark \
	--model=$MODEL\
  --model-url="http://localhost:8002/v1" \
  --dataset="offendo/math-atlas" \
  --output ./outputs/$OUTPUT_PATH \
  --max-tokens=8192 \
  --temperature=0.2 \
  --top-p=0.95 \
  --seed=1234 \
  --item-type=$SPLIT
