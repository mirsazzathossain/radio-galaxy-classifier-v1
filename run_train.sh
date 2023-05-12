#!/bin/bash
echo "Running training script..."
python utils/setup_configs.py --config downstream

for i in $(seq 1 1 6)
do
    echo "Now training model $i of 6..."
    python train.py --model downstream
    echo "Model $i of 6 trained!"
done

echo "Training complete!"