# data config
data_path = "../../data"
#dataset = "batch19"
dataset = "batch20"
log_dir = "/search/data/imer/fengchaobing/LSTM/test2/batch19_es256_hs1024/log"
model_dir = "/search/data/imer/fengchaobing/LSTM/test2/batch19_es256_hs1024/model/batch19_high20_384_low128_hs1024_model"
#model_dir = "/search/data/imer/fengchaobing/LSTM/test2/batch19_es256_hs1024/model/batch19_es512_hs1024_model"
model_name = "dynamic_lstm_1layer_embed256_hid1024_batch19_model"

# model config
num_layer = 1
#embedding_size = 256
#embedding_size = 512
embedding_size = 384
#embedding_size = 128
hidden_size = 1024

# train config
#vocab_size = 13761
vocab_size = 13639
batch_size = 512
max_len = 8
step_per_log = 200
step_per_validation =  46000
top_k = 3
lstm_keep_prob = 0.9
input_keep_prob = 0.9
output_keep_prob = 0.9
max_grad_norm = 5.0

