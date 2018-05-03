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
import json
import os
import threading
import time

from mock import patch
import urllib3

import sagemaker_containers as smc


class DummyTransformer(object):
    def __init__(self):
        self.calls = dict(initialize=0, transform=0)

    def initialize(self):
        self.calls['initialize'] += 1

    def transform(self):
        self.calls['transform'] += 1
        return smc.worker.TransformSpec(json.dumps(self.calls), smc.content_types.APPLICATION_JSON)


app = smc.worker.run(DummyTransformer(), module_name='app')


# class TestThread(threading.Thread):
#
#     def __init__(self, name='TestThread'):
#         super(self, TestThread).__init__(name)
#         self.event = threading.Event()
#
#     def run(self):
#         smc.server.start('test.functional.test_server_with_worker:app')
#
#     def join(self, timeout=None):
#         """ Stop the thread. """
#         self._stopevent.set()
#         threading.Thread.join(self, timeout)


@patch.dict(os.environ, {smc.environment.FRAMEWORK_MODULE_ENV: 'test.functional.test_server_with_worker:app',
                         smc.environment.USE_NGINX_ENV: 'false'})
def test_server():
    def start_server(stop_event):
        smc.server.start('test.functional.test_server_with_worker:app')

    pill2kill = threading.Event()
    t = threading.Thread(target=start_server, args=(pill2kill))
    t.start()

    time.sleep(2)

    http = urllib3.PoolManager()
    base_url = 'http://127.0.0.1:8080'
    r = http.request('GET', '{}/ping'.format(base_url))
    assert r.status == 200

    r = http.request('POST', '{}/invocations'.format(base_url))
    assert r.status == 200
    assert r.data.decode('utf-8') == dict(initialize=1, transform=1)

    pill2kill.set()
    t.join()
