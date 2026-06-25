from benchmark.utils.parameter import DatasetParameters, KQGRDatasetParams
from pydantic import Field, model_validator, ConfigDict
from typing import List, Literal, Union
from pathlib import Path

class TemporalDatasetParams(DatasetParameters):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    name: Literal["temporal"] = "temporal"
    init: str = Field('Temporal', json_schema_extra={'expand': False})
    dimensions: Union[int, List[int]] = Field(1, description="Number of independent bit streams")
    task: str = Field('density', description="two options: density or parity")
    bits: Union[int, List[int]] = Field(10, description="length of the bit stream")
    window: Union[int, List[int]] = Field(5, description="size of the window for temporal data")
    delay: Union[int, List[int]] = Field(0, description="delay; shifts window from right to left; higher delay is easier as bits are processed right to left")
    sampling_mode: Union[str, List[str]] = Field( 'random',
    description="'random': random bit patterns with repetition allowed. "
                "'exhaustive': enumerate patterns 0 to 2^bits-1, taking first 'samples' patterns (or cycling if samples > 2^bits)"
    )


    @model_validator(mode='after')
    def update_path_after_init(self):
        self.path = self._generate_path()
        return self

    def _generate_path(self):
        if self.has_list_in_a_field(): # not yet expanded (doesnt check recursively)
            return

        return (Path('data/temporal')
            / self.task
            / f'{self.dimensions}D'
            / f's-{self.sampling_mode}'
            / f'b-{self.bits}'
            / f'w-{self.window}'
            / f'd-{self.delay}'
            / self._base_data_path()
            / 'dataset.npz'
        )
    
    def update_path(self):
        self.path = self._generate_path()

    @property
    def init_obj(self):
        from project.temporal.code.dataset_init import TemporalDatasetInit
        return TemporalDatasetInit()


class KQGRTemporalDatasetParams(KQGRDatasetParams, TemporalDatasetParams):
    """
    Temporal dataset augmented with KQGR fields (tau, evaluation).

    Hijacks TemporalDensityDataset for data generation but only uses x values (input
    bitstreams) - y values (labels) are ignored since KQ/GR metrics evaluate reservoir
    kernel quality and generalization capability, not task performance.

    tau/evaluation_mode control GR metric: during encoding, tau bits per feature are set
    identical across all samples to test generalization under reduced input diversity.
    """