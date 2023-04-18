# -*- coding: utf-8 -*-

"""
BYOL Implementation in PyTorch.

This code has been adapted from: https://github.com/lucidrains/byol-pytorch
"""

__author__ = "Mir Sazzat Hossain"

import copy
import random
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as T


def default(val: any, def_val: any) -> any:
    """
    Return default value if val is None.

    :param val: value
    :type val: any
    :param def_val: default value
    :type def_val: any

    :return: value or default value
    :rtype: any
    """
    return def_val if val is None else val


def flatten(t: torch.Tensor) -> torch.Tensor:
    """
    Flatten tensor.

    :param t: tensor
    :type t: torch.Tensor

    :return: flattened tensor
    :rtype: torch.Tensor
    """
    return t.reshape(t.shape[0], -1)


def singleton(cache_key: str) -> any:
    """
    Singleton decorator.

    :param cache_key: cache key
    :type cache_key: str

    :return: singleton
    :rtype: any
    """
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def get_module_device(module: nn.Module) -> torch.device:
    """
    Get module device.

    :param module: module
    :type module: nn.Module

    :return: device
    :rtype: torch.device
    """
    return next(module.parameters()).device


def set_requires_grad(model: nn.Module, val: bool) -> None:
    """
    Set requires grad.

    :param model: model
    :type model: nn.Module
    :param val: value
    :type val: bool

    :return: None
    :rtype: None
    """
    for p in model.parameters():
        p.requires_grad = val


def loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Loss function.

    :param x: tensor
    :type x: torch.Tensor
    :param y: tensor
    :type y: torch.Tensor

    :return: loss
    :rtype: torch.Tensor
    """
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# augmentation utils
class RandomApply(nn.Module):
    """Random Apply."""

    def __init__(
        self,
        fn: object,
        p: float
    ) -> None:
        """
        Initialize Random Apply.

        :param fn: transformation function
        :type fn: object
        :param p: probability
        :type p: float
        """
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param x: tensor
        :type x: torch.Tensor

        :return: transformed tensor
        :rtype: torch.Tensor
        """
        if random.random() > self.p:
            return x
        return self.fn(x)


class EMA:
    """Exponential Moving Average."""

    def __init__(self, beta: float) -> None:
        """
        Initialize EMA.

        :param beta: beta
        :type beta: float
        """
        super().__init__()
        self.beta = beta

    def update_average(
        self,
        old: torch.Tensor,
        new: torch.Tensor
    ) -> torch.Tensor:
        """
        Update average.

        :param old: old tensor
        :type old: torch.Tensor
        :param new: new tensor
        :type new: torch.Tensor

        :return: updated tensor
        :rtype: torch.Tensor
        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(
    ema_updater: EMA,
    ma_model: nn.Module,
    current_model: nn.Module
) -> None:
    """
    Update moving average.

    :param ema_updater: EMA updater
    :type ema_updater: EMA
    :param ma_model: moving average model
    :type ma_model: nn.Module
    :param current_model: current model
    :type current_model: nn.Module
    """
    for current_params, ma_params in zip(
        current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def MLP(
    dim: int,
    projection_size: int,
    hidden_size: int = 4096
) -> nn.Sequential:
    """
    Multi-layer perceptron.

    :param dim: dimension
    :type dim: int
    :param projection_size: projection size
    :type projection_size: int
    :param hidden_size: hidden size
    :type hidden_size: int

    :return: MLP
    :rtype: nn.Sequential
    """
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )


def SimSiamMLP(
    dim: int,
    projection_size: int,
    hidden_size: int = 4096
) -> nn.Sequential:
    """
    MLP for SimSiam.

    :param dim: dimension
    :type dim: int
    :param projection_size: projection size
    :type projection_size: int
    :param hidden_size: hidden size
    :type hidden_size: int

    :return: SimSiam MLP
    :rtype: nn.Sequential
    """
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False),
    )


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets
class NetWrapper(nn.Module):
    """
    Wrapper class for the base neural network.

    It will manage the interception of the hidden layer output
    and pipe it into the projecter and predictor nets
    """

    def __init__(
        self,
        net: nn.Module,
        projection_size: int,
        projection_hidden_size: int,
        layer: any = -2,
        use_simsiam_mlp: bool = False
    ) -> None:
        """
        Initialize NetWrapper.

        :param net: base neural network
        :type net: nn.Module
        :param projection_size: projection size
        :type projection_size: int
        :param projection_hidden_size: projection hidden size
        :type projection_hidden_size: int
        :param layer: layer
        :type layer: Union[str, int]
        :param use_simsiam_mlp: use SimSiam MLP
        :type use_simsiam_mlp: bool
        """
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self) -> nn.Module:
        """Find layer."""
        if type(self.layer) is str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)  # skipcq: PTC-W0039
        if type(self.layer) is int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _: any, input: any, output: any) -> None:
        """Create hook."""
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self) -> None:
        """Register hook."""
        layer = self._find_layer()
        if layer is None:
            raise AssertionError(f"hidden layer ({self.layer}) not found")
        _ = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton("projector")
    def _get_projector(self, hidden: torch.Tensor) -> nn.Module:
        """
        Get projector.

        :param hidden: hidden layer
        :type hidden: torch.Tensor

        :return: projector
        :rtype: nn.Module
        """
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(
            dim, self.projection_size, self.projection_hidden_size
        )
        return projector.to(hidden)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get representation.

        :param x: input
        :type x: torch.Tensor

        :return: representation
        :rtype: torch.Tensor
        """
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()

        if hidden is None:
            raise AssertionError(
                f"hidden layer {self.layer} never emitted an output")
        return hidden

    def forward(
        self,
        x: torch.Tensor,
        return_projection: bool = True
    ) -> any:
        """
        Forward pass.

        :param x: input
        :type x: torch.Tensor
        :param return_projection: return projection
        :type return_projection: bool

        :return: projection, representation
        :rtype: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        """
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation


# main class
class BYOL(nn.Module):
    """Bootstrap Your Own Latent (BYOL) implementation."""

    def __init__(
        self,
        net: nn.Module,
        image_size: int,
        hidden_layer: any = -2,
        projection_size: int = 256,
        projection_hidden_size: int = 4096,
        augment_fn: object = None,
        augment_fn2: object = None,
        moving_average_decay: float = 0.99,
        use_momentum: bool = True,
    ) -> None:
        """
        Initialize BYOL.

        :param net: base neural network
        :type net: nn.Module
        :param image_size: image size
        :type image_size: int
        :param hidden_layer: hidden layer
        :type hidden_layer: Union[str, int]
        :param projection_size: projection size
        :type projection_size: int
        :param projection_hidden_size: projection hidden size
        :type projection_hidden_size: int
        :param augment_fn: augmentation function
        :type augment_fn: object
        :param augment_fn2: augmentation function 2
        :type augment_fn2: object
        :param moving_average_decay: moving average decay
        :type moving_average_decay: float
        :param use_momentum: use momentum
        :type use_momentum: bool
        """
        super().__init__()
        self.net = net

        # default SimCLR augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
            T.RandomResizedCrop((image_size, image_size)),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, self.augment1)

        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            layer=hidden_layer,
            use_simsiam_mlp=not use_momentum,
        )

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(
            projection_size, projection_size, projection_hidden_size
        )

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 1, image_size, image_size, device=device))

    @singleton("target_encoder")
    def _get_target_encoder(self) -> NetWrapper:
        """
        Get target encoder.

        :return: target encoder
        :rtype: NetWrapper
        """
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self) -> None:
        """Reset moving average."""
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self) -> None:
        """Update moving average."""
        if not self.use_momentum:
            raise AssertionError(
                "you do not need to update the moving average, since you \
                    have turned off momentum for the target encoder"
            )
        if self.target_encoder is None:
            raise AssertionError("target encoder has not been created yet")
        update_moving_average(
            self.target_ema_updater, self.target_encoder, self.online_encoder
        )

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
        return_projection: bool = True
    ) -> any:
        """
        Forward pass.

        :param x: input
        :type x: torch.Tensor
        :param return_embedding: return embedding
        :type return_embedding: bool
        :param return_projection: return projection
        :type return_projection: bool

        :return: projection, representation
        :rtype: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        """
        if self.training and x.shape[0] == 1:
            raise AssertionError(
                "you must have greater than 1 sample when training, due to \
                    the batchnorm in the projection layer"
            )

        if return_embedding:
            return self.online_encoder(x, return_projection=return_projection)

        image_one, image_two = self.augment1(x), self.augment2(x)

        online_proj_one, _ = self.online_encoder(image_one)
        online_proj_two, _ = self.online_encoder(image_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = (
                self._get_target_encoder() if self.use_momentum
                else self.online_encoder
            )
            target_proj_one, _ = target_encoder(image_one)
            target_proj_two, _ = target_encoder(image_two)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()
