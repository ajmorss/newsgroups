some_command &
P1=$!
other_command &
P2=$!
wait $P1 $P2

python trainer.py \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2223,localhost:2221 \
     --job_name=ps --task_index=0
# On worker0.example.com:
python trainer.py \
     --ps_hosts=ps0.example.com:2222 \
     --worker_hosts=localhost:2223,localhost:2221 \
     --job_name=worker --task_index=0
# On worker1.example.com:
python trainer.py \
     --ps_hosts=ps0.example.com:2222 \
     --worker_hosts=localhost:2223,localhost:2221 \
     --job_name=worker --task_index=1

