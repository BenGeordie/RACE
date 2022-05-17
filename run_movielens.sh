thresholds=(0.06 0.058 0.056 0.054 0.052 0.05 0.048 0.046 0.044 0.042)
gpu_idxs=(3 3 4 4 5 5 6 6 7 7)

echo "Movielens end-to-end. thresh = ${thresholds[$1]}"
echo "Using gpu ${gpu_idxs[$1]}"
python3 /home/bg31/RACE/main_train_with_race.py --data movielens --h 400 --n 3 --thrsh ${thresholds[$1]} --prob 0.1 --gpu ${gpu_idxs[$1]} --eval_step 300
