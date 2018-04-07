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

import os

from sagemaker_containers.environment import Environment


# TODO (mvs) - create shortcut decorators for public methods, e.g.: env.TrainingEnvironment().model_dir -> env.model_dir
class TrainingEnvironment(Environment):
    """Provides access to aspects of the training environment relevant to training jobs, including
    hyperparameters, system characteristics, filesystem locations, environment variables and configuration settings.

    Example on how a script can use training environment:
        ```
        import os
        import numpy as np

        import sagemaker_containers.training_environment as env

        from keras.applications.resnet50 import ResNet50

        # get the path of the channel 'training' from the inputdataconfig.json file
        training_dir = env.channel_input_dirs['training']

        # get a the hyperparameter 'training_data_file' from hyperparameters.json file
        file_name = hyperparameters['training_data_file']

        # get the folder where the model should be saved
        model_dir = env.model_dir

        data = np.load(os.path.join(training_dir, training_data_file))

         x_train, y_train = data['features'], keras.utils.to_categorical(data['labels'])

        model = ResNet50(weights='imagenet')

        # unfreeze the model to allow fine tuning
        ...

        model.fit(x_train, y_train)

        # save the model in the end of training
        model.save(os.path.join(model_dir, 'saved_model'))
        ```
    """

    def __init__(self, base_dir=None):
        """Construct an `TrainingEnvironment` instance.

        Args:
            base_dir (string): the base directory where the environment will read/write files.
                The default base directory in SageMaker is '/opt/ml'
        """
        super(TrainingEnvironment, self).__init__(base_dir)

        self._input_dir = os.path.join(self.base_dir, 'input')
        self._input_config_dir = os.path.join(self._input_dir, 'config')
        self._input_data_dir = os.path.join(self._input_dir, 'data')
        self._output_dir = os.path.join(self.base_dir, 'output')
        self._output_data_dir = os.path.join(self._output_dir, 'data')

    @property
    def input_dir(self):
        """The input_dir, e.g. /opt/ml/input/, is the directory where SageMaker saves input data
        and configuration files before and during training.

        The input data directory has the following subdirectories:
            config (`input_config_dir`) and data (`input_data_dir`)

        Returns:
            (string): the path of the input directory, e.g. /opt/ml/input/
        """
        return self._input_dir

    @property
    def input_config_dir(self):
        """The directory where standard SageMaker configuration files are located, e.g. /opt/ml/input/config/.

        SageMaker training creates the following files in this folder when training starts:
            - `hyperparameters.json`: Amazon SageMaker makes the hyperparameters in a CreateTrainingJob
                request available in this file.
            - `inputdataconfig.json`: You specify data channel information in the InputDataConfig parameter
                in a CreateTrainingJob request. Amazon SageMaker makes this information available
                in this file.
            - `resourceconfig.json`: name of the current host and all host containers in the training

        More information about this files can be find here:
            https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html

        Returns:
            (string): the path of the input directory, e.g. /opt/ml/input/config/
        """
        return self._input_config_dir

    @property
    def input_data_dir(self):
        """The directory where standard SageMaker configuration files are located.

        The input_data_dir, e.g. /opt/ml/input/data/, is the directory where SageMaker saves input data
        before and during training.

        The input data directory has the following subdirectories:
            config (`input_config_dir`) and data (`input_data_dir`)

        Returns:
            (string): the path of the input directory, e.g. /opt/ml/input/data/
        """
        return self._input_data_dir

    @property
    def output_dir(self):
        """The directory where training success/failure indications will be written, e.g. /opt/ml/output.

        To save non-model artifacts check `output_data_dir`.

        Returns:
            (string): the path to the output directory, e.g. /opt/ml/output/.
        """
        return self._output_dir

    @property
    def output_data_dir(self):
        """The dir to write non-model training artifacts (e.g. evaluation results) which will be retained
        by SageMaker, e.g. /opt/ml/output/data.

        As your algorithm runs in a container, it generates output including the status of the
        training job and model and output artifacts. Your algorithm should write this information
        to the this directory.

        Returns:
            (string): the path to output data directory, e.g. /opt/ml/output/data.
        """
        return self._output_data_dir

    @property
    def resource_config(self):
        """A dict<string, string> with the contents from /opt/ml/input/config/resourceconfig.json.

        It has the following keys:
            - current_host: The name of the current container on the container network.
                For example, 'algo-1'.
            -  hosts: The list of names of all containers on the container network, sorted lexicographically.
                For example, `["algo-1", "algo-2", "algo-3"]` for a three-node cluster.

        Returns:
            dict<string, string>
        """
        # TODO (mvs) - load resource config from file
        return {'current_host': 'algo-1', 'hosts': ['algo-1']}

    @property
    def hyperparameters(self):
        """The dict of hyperparameters that were passed to the training job.

        Returns:
            dict[string, object]: the hyperparameters.
        """
        # TODO (mvs) - load hyperparameters from file
        return {}

    @property
    def current_host(self):
        """The name of the current container on the container network. For example, 'algo-1'.

        Returns:
            string: current host.
        """
        return self.resource_config['current_host']

    @property
    def hosts(self):
        """The list of names of all containers on the container network, sorted lexicographically.
                For example, `["algo-1", "algo-2", "algo-3"]` for a three-node cluster.

        Returns:
              list[string]: all the hosts in the training network.
        """
        return self.resource_config['hosts']

    @property
    def input_data_config(self):
        """A dict<string, string> with the contents from /opt/ml/input/config/inputdataconfig.json.

        For example, suppose that you specify three data channels (train, evaluation, and validation) in
        your request. This dictionary will contain:

        ```{"train": {
                "ContentType":  "trainingContentType",
                "TrainingInputMode": "File",
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None"
            },
            "evaluation" : {
                "ContentType": "evalContentType",
                "TrainingInputMode": "File",
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None"
            },
            "validation": {
                "TrainingInputMode": "File",
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None"
            }
         } ```

        You can find more information about /opt/ml/input/config/inputdataconfig.json here:
            https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-inputdataconfig

        Returns:
            dict<string, string>
        """
        # TODO (mvs) - load config from file
        return {}

    @property
    def channel_input_dirs(self):
        """A dict[string, string] containing the data channels and the directories where the training
        data was saved.

        When you run training, you can partition your training data into different logical "channels".
        Depending on your problem, some common channel ideas are: "train", "test", "evaluation"
            or "images',"labels".

        The format of channel_input_dir is as follows:

            - `channel`[key] - the name of the channel defined in the input_data_config.
            - `training data path`[value] - the path to the directory where the training data is saved.

        Returns:
            dict[string, string] with the information about the channels.
        """
        # TODO (mvs) - load config from file
        return {}

    @property
    def user_script_name(self):
        """Returns the user script name if a user script was submitted for training.

        Returns:
              string: user script name.
        """
        # TODO (mvs) - implementation
        return 'user_script'

    def write_success_file(self, message=None):
        """Saves a success file that will be used by SageMaker to report success.

        If training succeed, your algorithm can optionally invoke this method to report the success to
        SageMaker by writing to this file.

        Args:
            message (string): the message to be written in the success file.
        """
        # TODO (mvs) - implementation
        pass

    def write_failure_file(self, message):
        """Saves a failure file that will be used by SageMaker to report failure.

        If training fails, your algorithm should write the failure description to this file.

        In a DescribeTrainingJob response, Amazon SageMaker returns the first 1024 characters from
        this file as FailureReason.

        Args:
            message (string): the message to be written in the failure file.
        """
        # TODO (mvs) - implementation
        pass
