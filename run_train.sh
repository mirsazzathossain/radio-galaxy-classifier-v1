#!/bin/bash
echo "Running training script..."
python utils/setup_configs.py --config downstream

for i in $(seq 1 1 15)
do
    echo "Now training model $i of 15..."
    python train.py --model downstream
    echo "Model $i of 15 trained!"
done

echo "Training complete!"