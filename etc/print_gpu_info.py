#!/usr/bin/env python
from __future__ import absolute_import
import subprocess

try:
    args = ["nvidia-smi", "--format=csv,noheader,nounits", "--query-gpu=utilization.gpu"]
    msg = subprocess.check_output(args)

    gpus = msg.decode("utf-8").strip().split('\n')

    msgs = ["gpu-{}={}".format(idx, val) for idx, val in enumerate(gpus)]

    fields = ','.join(msgs)
    print('gpu_utilization {}'.format(fields))
except OSError:
    pass
