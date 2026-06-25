from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Optional, Union, List, Literal
from pathlib import Path
import hashlib
import json


class KQGRDatasetParams(BaseModel):
    """
    Mixin adding KQGR capacity metric fields to any dataset type.
    tau has no default — configs without tau fail validation for KQGRXxx classes
    so Pydantic discriminates automatically based on tau presence.
    """
    tau: Union[int, List[int]]
    tau_axis: Literal['resolution', 'steps'] = Field(
        'resolution',
        description="Axis along which tau identical inputs are injected for GR measurement. "
                    "'resolution': tau reduces resolution, injected along the b-dimension. "
                    "'steps': tau appends identical steps along the s-dimension."
    )
    tau_mode: Literal['steal', 'augment'] = Field(
        'steal',
        description="steal: reduce free bits by tau (default); "
                    "augment: inflate baseline by tau then steal, keeping all resolution bits free "
                    "so 2^resolution distinct inputs are available regardless of tau."
    )
    evaluation: Union[str, List[str]] = Field('last', description="Tau application mode: 'first', 'last', or 'random'")

    @model_validator(mode='before')
    @classmethod
    def _assert_no_field_clash(cls, values):
        kqgr_fields = set(KQGRDatasetParams.model_fields.keys())
        for parent in cls.__mro__:
            if not hasattr(parent, 'model_fields'):
                continue
            if issubclass(parent, KQGRDatasetParams) or parent is BaseModel:
                continue  # skip KQGR classes (they own these fields) and BaseModel
            clash = kqgr_fields & set(parent.model_fields.keys())
            assert not clash, (
                f"KQGRDatasetParams field clash in {parent.__name__}: {clash}. "
                "Rename the conflicting field in the base dataset class."
            )
        return values

    @property
    def evaluation_mode(self) -> str:
        """First segment of evaluation string (e.g. 'last-random' → 'last')."""
        return str(self.evaluation).split('-')[0]


class Split(BaseModel):
    train: float = Field(0.8, description="data set fraction for training")
    dev: float = Field(0.1, description="data set fraction for development")
    test: float = Field(0.1, description="data set fraction for testing")

    @model_validator(mode='after')
    def must_sum_to_one(cls, values):
        total = float(sum((values.train, values.dev, values.test)))
        if total != 1.0:
            raise ValueError('The sum of train, dev, and test must be 1. Got {}'.format(total))
        return values

class DatasetParameters(BaseModel):
    name: str = Field(description="Discriminator for the dataset type")
    path: Optional[Path] = Field(None, description="Path to dataset")
    split: Split = Field(Split(), description="fraction for train, dev, test")
    shuffle: bool = Field(True, description="Shuffle dataset before splitting")
    generate_data: bool = Field(False, description="Ignores loading even if dataset exists at path")
    samples: int = Field(64, description="Number of samples to generate in the dataset")
    seed: Optional[int] = Field(None, description="Random seed, None disables seed")
    reset: bool = Field(True, description="If False, each sample's origin is the final position of the previous sample")
    init: str = Field('', json_schema_extra={'expand': False})

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='forbid'
    )

    def has_list_in_a_field(self):
        return any(isinstance(value, list) for value in self.model_dump().values())

    def _base_data_path(self) -> Path:
        """Path segments for data-affecting base parameters."""
        return Path(f'm-{self.samples}') / f'r-{self.seed}' / f'rst-{int(self.reset)}'

    @staticmethod
    def _hash_dict(data: dict, n_chars: int = 5) -> str:
        """SHA-256 hash of a plain dict. Use for uniqueness guarantees in path generation."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:n_chars]

    @staticmethod
    def _hash_params(params, n_chars=5):
        """Hash a Pydantic model's fields for path generation."""
        return DatasetParameters._hash_dict(params.model_dump(), n_chars)
