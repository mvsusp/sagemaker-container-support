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

import pytest as pytest

import sagemaker_containers as smc


@pytest.mark.parametrize('fn, expected', [
    (lambda: None, ([], None, None)),
    (lambda x, y='y': None, (['x', 'y'], None, None)),
    (lambda *args: None, ([], 'args', None)),
    (lambda **kwargs: None, ([], None, 'kwargs')),
    (lambda x, y, *args, **kwargs: None, (['x', 'y'], 'args', 'kwargs'))
])
def test_signature(fn, expected):
    assert smc.functions.signature(fn) == expected


@pytest.mark.parametrize('fn, environment, expected', [
    (lambda: None, {}, {}),
    (lambda x, y='y': None, dict(x='x', y=None, t=3), dict(x='x', y=None)),
    (lambda *args: None, dict(x='x', y=None, t=3), {}),
    (lambda **kwargs: None, dict(x='x', y=None, t=3), dict(x='x', y=None, t=3))
])
def test_matching_args(fn, environment, expected):
    assert smc.functions.matching_args(fn, environment) == expected
