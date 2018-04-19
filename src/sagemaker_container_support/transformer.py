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

import collections
from functools import partial

import sagemaker_container_support as scs


Framework = collections.namedtuple('Framework', ['model_fn', 'input_fn', 'predict_fn', 'output_fn'])


class Transformer(object):
    def __init__(self):
        self._model = None
        self._call = None

    def from_functions(self, user_module, model_fn=None, input_fn=None, predict_fn=None, output_fn=None):
        framework = Framework(model_fn=model_fn, input_fn=input_fn, predict_fn=predict_fn, output_fn=output_fn)

        self._call = partial(scs.functions.call, user_module=user_module, framework=framework)

    def initialize(self, env):
        self._model = self._call(fn_name='model_fn', model_dir=env.model_dir)

    def transform(self, request_env):
        return self._call(fn_name='transform_fn', data=request_env.data,
                          content_type=request_env.content_type,
                          model=request_env.model, accept=request_env.accept)

    def transform_fn(self, model, data, content_type, accept):
        input_data = self._call(fn_name='input_fn', data=data, content_type=content_type)

        prediction = self._call(fn_name='predict_fn', model=model, input_data=input_data)

        return self._call(fn_name='output_fn', prediction=prediction, accept=accept)
