from torch import nn
from abc import abstractmethod
from typing import List, Any
from torch import Tensor
class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor, isnan_mask: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError


    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, input: Any, **kwargs) -> Tensor:
        pass



