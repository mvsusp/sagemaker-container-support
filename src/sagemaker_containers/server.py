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

import subprocess

import gunicorn.app.base
import pkg_resources

import sagemaker_containers as smc

UNIX_SOCKET_BIND = 'unix:/tmp/gunicorn.sock'
HTTP_BIND = '0.0.0.0:8080'


class GunicornApp(gunicorn.app.base.BaseApplication):
    """Standalone gunicorn application

    """

    def __init__(self, app, timeout=None, worker_class=None, bind=None,
                 worker_connections=None, workers=None, log_level=None):
        self.log_level = log_level
        self.workers = workers
        self.worker_connections = worker_connections
        self.bind = bind
        self.worker_class = worker_class
        self.timeout = timeout
        self.application = app
        super(GunicornApp, self).__init__()

    def load_config(self):
        self.cfg.set('timeout', self.timeout)
        self.cfg.set('worker_class', self.worker_class)
        self.cfg.set('bind', self.bind)
        self.cfg.set('worker_connections', self.worker_connections)
        self.cfg.set('workers', self.workers)
        self.cfg.set('loglevel', self.log_level)

    def load(self):
        return self.application


def start(app):
    env = smc.environment.ServingEnvironment()
    gunicorn_bind_address = HTTP_BIND
    processes = []

    if env.use_nginx:
        gunicorn_bind_address = UNIX_SOCKET_BIND
        nginx_config_file = pkg_resources.resource_filename(smc.__name__, '/etc/nginx.conf')
        nginx = subprocess.Popen(['nginx', '-c', nginx_config_file])
        processes.append(nginx)

    try:
        GunicornApp(app=app, timeout=env.model_server_timeout, worker_class='gevent',
                    bind=gunicorn_bind_address, worker_connections=1000 * env.model_server_workers,
                    workers=env.model_server_workers, log_level='debug').run()
    finally:
        for p in processes:
            p.terminate()
