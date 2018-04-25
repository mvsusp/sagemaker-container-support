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

import inspect

import six

import sagemaker_containers as smc


def matching_args(fn, dictionary):  # type: (function, collections.Mapping) -> dict
    """Given a function fn and a dict dictionary, returns the function arguments that match the dict keys.

    Example:

        def train(channel_dirs, model_dir): pass

        dictionary = {'channel_dirs': {}, 'model_dir': '/opt/ml/model', 'other_args': None}

        args = smc.functions.matching_args(train, dictionary) # {'channel_dirs': {}, 'model_dir': '/opt/ml/model'}

        train(**args)
    Args:
        fn (function): a function
        dictionary (dict): the dictionary with the keys

    Returns:
        (dict) a dictionary with only matching arguments.
    """
    args, _, kwargs = signature(fn)

    if kwargs:
        return dictionary

    return smc.collections.split_by_criteria(dictionary, set(args))[0]


def signature(fn):  # type: (function) -> ([], [], [])
    """Given a function fn, returns the function args, vargs and kwargs

    Args:
        fn (function): a function

    Returns:
        ([], [], []): a tuple containing the function args, vargs and kwargs.
    """
    if six.PY2:
        arg_spec = inspect.getargspec(fn)
        return arg_spec.args, arg_spec.varargs, arg_spec.keywords
    elif six.PY3:
        sig = inspect.signature(fn)

        def filter_parameters(kind):
            return [
                p.name for p in sig.parameters.values()
                if p.kind == kind
            ]

        args = filter_parameters(inspect.Parameter.POSITIONAL_OR_KEYWORD)

        vargs = filter_parameters(inspect.Parameter.VAR_POSITIONAL)

        kwargs = filter_parameters(inspect.Parameter.VAR_KEYWORD)

        return (args,
                vargs[0] if vargs else None,
                kwargs[0] if kwargs else None)
