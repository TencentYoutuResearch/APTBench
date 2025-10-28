export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# node 1
python3 -m sglang.launch_server --model-path $1 --served-model-name $2 --tp 16 --dist-init-addr $3 --nnodes 2 --node-rank 0 --trust-remote-code --host 0.0.0.0 --port 8001
