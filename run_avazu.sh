thresholds=(0.015 0.0145 0.014 0.0135 0.013 0.0125 0.012 0.0115 0.011 0.0105)

echo "Avazu end-to-end. thresh = ${thresholds[$1]}"
echo "Using gpu $2"
/usr/bin/python3 /home/bg31/RACE/main_train_with_race.py --data avazu --thrsh ${thresholds[$1]} --prob 0.1 --gpu $2
