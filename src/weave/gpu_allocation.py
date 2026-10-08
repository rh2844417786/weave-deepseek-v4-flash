"""Use the scheduler's four physical GPU indices at every host boundary."""
from dataclasses import dataclass
import os
import re


@dataclass(frozen=True)
class GPUAllocation:
    ids: tuple

    def __post_init__(self):
        if (len(self.ids)!=4 or len(set(self.ids))!=4
                or any(type(value) is not int or value<0 for value in self.ids)):
            raise ValueError('CUDA_VISIBLE_DEVICES must contain four distinct nonnegative GPU indices')

    @classmethod
    def parse(cls,value):
        if not isinstance(value,str) or not value.strip():
            raise ValueError('GPUQ must supply CUDA_VISIBLE_DEVICES with four distinct GPU indices')
        fields=[part.strip() for part in value.split(',')]
        if any(re.fullmatch(r'[0-9]+',part) is None for part in fields):
            raise ValueError('CUDA_VISIBLE_DEVICES must contain four distinct nonnegative GPU indices')
        return cls(tuple(int(part) for part in fields))

    @property
    def csv(self): return ','.join(str(value) for value in self.ids)

    @property
    def docker_argument(self): return f'"device={self.csv}"'

    @property
    def container_visible(self): return ','.join(str(value) for value in range(len(self.ids)))


def allocated_gpus():
    return GPUAllocation.parse(os.environ.get('CUDA_VISIBLE_DEVICES'))
