import gym
import numpy as np

from typing import Optional
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import List, Dict, TensorType, ModelConfigDict
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.preprocessors import get_preprocessor

torch, nn = try_import_torch()


class CustomTorchModel(nn.Module, TorchModelV2):
    
    def __init__(self,
                obs_space: gym.spaces.Space,
                action_space: gym.spaces.Space,
                num_outputs: int,
                model_config: ModelConfigDict,
                name: str,
                **customized_model_kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))

        input_size = get_preprocessor(obs_space)(obs_space).size
        output_size = get_preprocessor(action_space)(action_space).size
        hidden_size = model_config["custom_model_config"]["fcnet_hiddens"]

        # add hidden layers
        for size in hidden_size:
            layers.append(
                SlimFC(
                    in_size         =   prev_layer_size,
                    out_size        =   size,
                    initializer     =   normc_initializer(1.0),
                    activation_fn   =   get_activation_fn(
                        model_config["custom_model_config"]["hidden_layer_activation"],
                        framework   =   "torch"
                    )
                )
            )
            prev_layer_size = size

            # add a batch norm layer
            layers.append(nn.BatchNorm1d(prev_layer_size))
        
        # add main policy action layer
        self._logits = SlimFC(
            in_size         =   prev_layer_size,
            out_size        =   output_size,
            initializer     =   normc_initializer(1.0),# 0.01),
            activation_fn   =   get_activation_fn(
                model_config["custom_model_config"]["action_layer_activation"],
                framework   =   "torch"
            )
        )

        # add value function layer
        self._value_branch = SlimFC(
            in_size         =   prev_layer_size,
            out_size        =   1,
            initializer     =   normc_initializer(1.0),
            activation_fn   =   get_activation_fn(
                model_config["custom_model_config"]["value_layer_activation"],
                framework   =   "torch"
            )
        )

        self._hidden_layers = nn.Sequential(*layers)
    
    @override(ModelV2)
    def forward(self, input_dict: Dict[str, TensorType], state: Optional[List[TensorType]] = [], seq_lens: Optional[TensorType] = None) -> tuple:
        if len(input_dict) == 1:
            train_eval_mode = False
        else:
            train_eval_mode = True
        
        self._hidden_layers.train(mode=train_eval_mode)
        self._hidden_out = self._hidden_layers(input_dict["obs"].float())

        # pass through main policy action layer
        logits = self._logits(self._hidden_out)

        return logits, state

    @override(ModelV2)
    def value_function(self) -> torch.tensor:
        assert self._hidden_out is not None, "must call forward first!"
        value_out = self._value_branch(self._hidden_out)
        
        return torch.reshape(value_out, [-1])
