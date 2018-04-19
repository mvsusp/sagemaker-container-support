# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import signal
import subprocess
import sys

import sagemaker_container_support.environment as env

NGINX_PID = 0


def add_terminate_signal(process):
    def terminate(signal_number, stack_frame):
        process.terminate()

    signal.signal(signal.SIGTERM, terminate)


def start(args):
    module_app = args[1]
    model_server_workers = env.get_model_server_workers()

    gunicorn_bind_address = '0.0.0.0:8080'

    nginx = None

    if env.use_nginx:
        subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
        subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])
        gunicorn_bind_address = 'unix:/tmp/gunicorn.sock'
        nginx = subprocess.Popen(['nginx', '-c', '/usr/local/etc/nginx.conf'])

        add_terminate_signal(nginx)

    gunicorn = subprocess.Popen(['gunicorn',
                                 '--timeout', str(env.model_server_timeout),
                                 '-k', 'gevent',
                                 '-b', gunicorn_bind_address,
                                 '--worker-connections', str(1000 * model_server_workers),
                                 '-w', str(model_server_workers),
                                 '--log-level', 'debug',
                                 module_app])

    add_terminate_signal(gunicorn)

    while True:
        if nginx and nginx.poll():
            gunicorn.terminate()
            break
        elif gunicorn.poll():
            nginx.terminate()
            break

    sys._exit(0)
