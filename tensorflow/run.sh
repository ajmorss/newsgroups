#some_command &
#P1=$!
#other_command &
#P2=$!
#wait $P1 $P2

python trainer.py \
     --distributed=True \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2223,localhost:2221 \
     --job_name=ps --task_index=0 \
     --config_yaml=config/sample.yaml 
# On worker0.example.com:
python trainer.py \
     --distributed=True \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2223,localhost:2221 \
     --job_name=worker --task_index=0 \
     --config_yaml=config/sample.yaml
# On worker1.example.com:
python trainer.py \
     --distributed=True \
     --ps_hosts=localhost:2222 \
     --worker_hosts=localhost:2223,localhost:2221 \
     --job_name=worker --task_index=1 \
     --config_yaml=config/sample.yaml

