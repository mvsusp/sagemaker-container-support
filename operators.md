
# SageMaker Operators

# Open MPI operator

## Horovod use case
Extracted from https://github.com/uber/horovod/blob/master/README.md#running-horovod.

1. To run on a machine with 4 GPUs:

```bash
$ mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```

2. To run on 4 machines with 4 GPUs each:

```bash
$ mpirun -np 16 \
    -H server1:4,server2:4,server3:4,server4:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```

### Using a entrypoint.sh script with env vars

The first example, using only one instance works fine by just using [our SageMaker Env Variables](https://github.com/mvsusp/sagemaker-containers/tree/master#sm_hosts):

**entrypoint.sh**
```bash
$ mpirun -np 4 \
    -H $SM_CURRENT_HOST:$SM_NUM_GPUS \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```

Both **entrypoint.sh** and **train.py** are under src:

```
src/
├── entrypoint.sh
└── train.py
```

Now we can create the Python SDK Estimator:

```python
estimator  = Estimator(entrypoint='entrypoint.sh', source_dir='src',...)
estimator.fit(...)
```


The second example **needs more than just env vars**. The script below would not even work given that:
- it should be executed only in one instance
- all other instances should wait the script to finish

```bash
# Not a simple script 

HOSTS_AND_GPUS=$(python <<EOF
import os
import json

hosts = json.loads(os.environ['SM_HOSTS'])
num_gpus = os.environ['SM_NUM_GPUS']
hosts_per_gpu = [host + ':' + num_gpus for host in hosts].join(',')

print(hosts_per_gpu)
EOF
)

$ mpirun -np 16 \
    -H ${HOSTS_AND_GPUS} \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```

### Using an entrypoint.sh with sm-mpirun operator

```
$ sm mpirun --help

sm mpirun - sagemaker operator for mpirun

Usage: sm mpirun [OPTIONS] <program>  [ <args> ] 

Options:
--process_per_hosts  

Integer that determines how many processes can  be run on each host. By default, this is equal to one process per host on CPU instances, or one process per GPU on GPU instances. 

--custom_mpi_options  

String of custom mpi options to pass to the mpirun command. There will some MPI options that are specific to sagemaker infrastructure and we would not allow customer to override that, rest can be overridden by customer. They can also pass in additional flags to this placeholder which will be appended to the MPI run command. See Appendix for the list of MPI flags classified in these categories.
```
Options extracted from [SageMaker Containers MPI Design](https://quip-amazon.com/aV4BAV4Eaofu#fCY9CAko4zg).

**entrypoint.sh**
```bash
# sm mpirun will execute:
# mpirun -np 4 \
#    -H localhost:4 \
#    -bind-to none -map-by slot \
#    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#    -mca pml ob1 -mca btl ^openib \
#    python train.py

sm mpirun python train.py
```
a more complete example would be:

**entrypoint.sh**
```bash
sm mpirun --process_per_hosts 3 --custom_mpi_options '-X TF_LOG_LEVEL=10' python train.py --learning-rate 10 
```

Operators can be used in Python as well:

**entrypoint.py**
```python
import sagemaker_containers.operators as sm

sm.mpirun(process_per_hosts=3, learning_rate=10)
```

#### sm mpirun source code
[Click](https://github.com/pallets/click#a-simple-example), one of the most used python modules for cli tools solves helps us to provide both python functions and cli binaries beautifully. 
```python
import click

@cli.command()
@click.option('--process_per_hosts', type=int)
@click.option('--custom_mpi_options')
@click.argument('program')  
@click.argument('args', nargs=-1, type=str)
def mpirun(program, 
           args, 
           process_per_hosts=None, 
           custom_mpi_options=''):
    ...
```

## Distributed TensorFlow use case
https://www.tensorflow.org/deploy/distributed
```bash
# On ps0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=0
# On ps1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=1
# On worker0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=0
# On worker1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=1

```

**entrypoint.sh**
```bash
sm tf run-distributed-training --num-workers 2 --num-parameter-servers 2 trainer.py
```

# Using with aws cli

```bash
root
├── Dockerfile
└── src
    ├── train.py
    └── entrypoint.sh
```
**Dockerfile**
```docker
FROM tensorflow:latest

pip install sagemaker-containers

ENTRYPOINT ['bash', 'entrypoint.sh']
```

```bash
docker build -t my-container ..
docker push my-container <ecs-tag>
aws sagemaker create-training-job ...
```
another option is to just override the entrypoint:

**horovod container**
```docker
FROM tensorflow:latest

pip install sagemaker-containers, horovod

ENTRYPOINT ['bash', 'sm', 'mpirun', 'python', 'train.py']
```

Customers can just override train.py:

```docker
FROM sagemaker-horovod-container

COPY my-training.py train.py
```

