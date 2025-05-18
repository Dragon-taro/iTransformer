export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u run.py \
  --is_training 0 \
  --root_path ./scripts/multivariate_forecasting/USDJPY/pretreatment/pretreatment_data/ \
  --data_path usdjpy_X_test_wdate.csv \
  --model_id usdjpy_60_10 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 60 \
  --pred_len 10 \
  --e_layers 2 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --target Close \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --itr 1 