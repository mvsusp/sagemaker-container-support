# Sagemaker Containers

## Creating SageMaker ML Frameworks

### Framework code
```python
import json

from sagemaker_containers import Training

def framework_train_fn(user_module, training_environment):
    model = user_module.train(training_environment.channel_dirs, training_environment.hyperparameters)
    save_model(model)

def save_model(model):
    model['saved'] = True
    with open('/opt/ml/output/saved_model.json', 'w') as f:
        json.dump(model, f)

if __name__ == '__main__':
    training = Training.from_train_fn(train_fn=framework_train_fn)
    training.start()
```

### User Script
```python
def train(chanel_dirs, hps): 
    return {'trained': True, 'saved': False}
```

## Using as a library
Users can provide a python package to execute in the container.

### User Module
**training_module/train.py**
```python
import sagemaker_containers.training_environment as env
import json
import os

def main():
    model = {'trained': True, 'saved': False}

    with open(os.path.join(env.model_dir, 'saved_model.json', 'w')) as f:
        json.dump(model, f)
        
if __name__ == '__main__':
    main()
```
### start training

Containers can use the function below to execute customers modules.  

```python
from sagemaker_containers import Training

if __name__ == '__main__':
    training = Training.from_module(module_name='customer_module.train')
    training.start()
```

### library API
```python
import sagemaker_containers.training_environment as env
```

```python
env.input_dir
```
The base directory for training data and configuration files: ```'/opt/ml/input'```

```python
env.input_config_dir
```
The directory where standard SageMaker configuration files are located: ```'/opt/ml/input/config'```

```python
env.output_dir
```
The base directory for training data and configuration files: ```'/opt/ml/input'```

```python
env.hyperparameters
```
The dict of hyperparameters that were passed to the CreateTrainingJob API: ```{'training_steps': 1000}```

```python
env.current_host
```
The hostname of the current container: ```'algo-1'```

```python
env.hosts
```
The list of hostnames available to the current training job: ```['algo-1', 'algo-2']```

```python
env.model_dir
```
```'/opt/ml/model'```

```python
env.code_dir
```
```'/opt/ml/code'```

```python
env.available_cpus
```
The number of cpus available in the current container: ```4```

```python
env.available_gpus
```
The number of gpus available in the current container: ```2```
