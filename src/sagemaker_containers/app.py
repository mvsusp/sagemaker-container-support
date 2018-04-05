# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from sagemaker_containers.training_engine import TrainingEngine


class App(object):
    def __init__(self):
        self.training_engine = TrainingEngine()

    def register_engine(self, engine):
        if isinstance(engine, TrainingEngine):
            self.training_engine = engine
        else:
            raise ValueError('Type: %s is not a valid engine type' % type(engine))

    def run(self):
        self.training_engine.run()
