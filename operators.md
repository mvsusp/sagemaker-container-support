
# SageMaker Operators

## Open MPI operator

### use cases

#### 1. Horovod
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

**generic bash solution for SageMaker**

The first example, using only one instance works fine by just using [our SageMaker Env Variables](https://github.com/mvsusp/sagemaker-containers/tree/master#sm_hosts):

```bash
$ mpirun -np 4 \
    -H $SM_CURRENT_HOST:$SM_NUM_GPUS \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```
The second example does not work so well. The script below would not even work given that:
- it should be executed only in one instance
- all other instances should wait the script to finish

```bash

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
