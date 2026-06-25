from typing import Union, List, Literal
from pydantic import Field, model_validator, ConfigDict
from pathlib import Path
import math
from benchmark.utils.parameter import DatasetParameters, KQGRDatasetParams
from benchmark.path_integration.constrained_foraging_path import (
    LevyFlightStrategy, SimpleRandomWalkStrategy, PhysicsWalkStrategy,
    PolygonBoundary, IntervalBoundary, NoBoundary,
    SoftBoundary, LinearSoftBoundary, QuadraticSoftBoundary, CubicSoftBoundary,
    generate_polygon_points, stretch_polygon
)
from project.boolean_reservoir.code.utils.param_utils import DynamicParams, ExpressionEvaluator

def strategy_factory(p: DynamicParams):
    """Factory function to create strategy objects"""
    strategy_map = {
        'LevyFlightStrategy': LevyFlightStrategy,
        'SimpleRandomWalkStrategy': SimpleRandomWalkStrategy,
        'PhysicsWalkStrategy': PhysicsWalkStrategy,
    }

    if p.name not in strategy_map:
        raise ValueError(f"Unsupported strategy type: {p.name}. Available: {list(strategy_map.keys())}")

    return p.call(strategy_map[p.name])
    
def boundary_factory(p: DynamicParams):
    """Factory function to create boundary objects"""
    evaluator = ExpressionEvaluator(symbols={'pi': math.pi})
    
    if p.name == 'PolygonBoundary':
        points = p.call(generate_polygon_points, evaluator=evaluator)
        boundary = PolygonBoundary(points=points)
        return p.call(stretch_polygon, evaluator=evaluator, boundary=boundary)
    
    elif p.name == 'IntervalBoundary':
        return p.call(IntervalBoundary)
    
    elif p.name == 'NoBoundary':
        return p.call(NoBoundary)

    elif p.name == 'LinearSoftBoundary':
        return p.call(LinearSoftBoundary)

    elif p.name == 'QuadraticSoftBoundary':
        return p.call(QuadraticSoftBoundary)

    elif p.name == 'CubicSoftBoundary':
        return p.call(CubicSoftBoundary)

    else:
        raise ValueError(f"Unsupported boundary type: {p.name}. Available: ['PolygonBoundary', 'IntervalBoundary', 'NoBoundary', 'LinearSoftBoundary', 'QuadraticSoftBoundary', 'CubicSoftBoundary']")


class PathIntegrationDatasetParams(DatasetParameters):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    name: Literal["path_integration"] = "path_integration"
    init: str = Field('PathIntegration', json_schema_extra={'expand': False})
    coordinate: Union[str, List[str]] = Field('cartesian', description='Coordinate system for inputs: cartesian or polar')
    output_coordinate: Union[str, List[str]] = Field('cartesian', description='Coordinate system for target output: cartesian or polar')
    mode: Union[str, List[str]] = Field('displacement', description="Input representation from optimistic trajectory: 'displacement' (sum=p_final), 'velocity', or 'acceleration' (a_ext+a_bnd_opt)")
    dimensions: Union[int, List[int]] = Field(2, description="Number of dimensions")
    steps: Union[int, List[int]] = Field(10, description="Number of steps")

    strategy: Union[DynamicParams, List[DynamicParams]] = Field(
        default=DynamicParams(
            name='SimpleRandomWalkStrategy',
        ),
        description="Strategy configuration"
    )

    boundary: Union[DynamicParams, List[DynamicParams]] = Field(
        default=DynamicParams(
            name='PolygonBoundary',
            params={'n_sides': 4, 'radius': 1.0, 'rotation': math.pi/4, 'stretch_x': 1.0, 'stretch_y': 1.0}
        ),
        description="Boundary configuration"
    )

    @model_validator(mode='after')
    def update_path_after_init(self):
        self.path = self._generate_path()
        return self

    @property
    def strategy_obj(self):
        """Property that constructs the strategy object using factory"""
        return strategy_factory(self.strategy)
    
    @property
    def boundary_obj(self):
        """Property that constructs the boundary object using factory"""
        return boundary_factory(self.boundary)

    @property
    def init_obj(self):
        from project.path_integration.code.dataset_init import PathIntegrationDatasetInit
        return PathIntegrationDatasetInit()
        
    def _generate_path(self) -> Path:
        """Generate path based on parameters"""
        if self.has_list_in_a_field(): # not yet expanded (doesnt check recursively)
            return

        strategy_str = self.strategy.name[:3].upper()  # First 3 chars of name
        strategy_hash = self._hash_params(self.strategy.params)
        resolution_bits = self.strategy.params.model_dump().get('resolution_bits', None)
        res_str = f'res-{resolution_bits}' if resolution_bits is not None else 'res-cont'

        boundary_str = self.boundary.name[:3].upper()  # First 3 chars of name
        boundary_hash = self._hash_params(self.boundary.params)

        # Full hash of all data-generating fields — catches any param not explicit in the path above
        full_hash = self._hash_dict({
            'coordinate': self.coordinate,
            'output_coordinate': self.output_coordinate,
            'dimensions': self.dimensions,
            'steps': self.steps,
            'mode': self.mode,
            'strategy': self.strategy.model_dump(),
            'boundary': self.boundary.model_dump(),
            'samples': self.samples,
            'seed': self.seed,
            'reset': self.reset,
        }, n_chars=8)

        return (
            Path('data/path_integration')
            / f'c-{self.coordinate}'
            / f'oc-{self.output_coordinate}'
            / f'd-{self.dimensions}'
            / f's-{self.steps}'
            / f'm-{self.mode}'
            / strategy_str
            / strategy_hash
            / res_str
            / boundary_str
            / boundary_hash
            / self._base_data_path()
            / full_hash
            / 'dataset.npz'
        )

    def update_path(self):
        """Update the path based on current parameters"""
        self.path = self._generate_path()


class KQGRPathIntegrationDatasetParams(KQGRDatasetParams, PathIntegrationDatasetParams):
    """PathIntegration dataset augmented with KQGR fields (tau, evaluation)."""

if __name__ == '__main__':
    import yaml
    
    yaml_content = """
    dimensions: [2, 3]
    steps: 10
    strategy:
      name: LevyFlightStrategy
      params:
        alpha: [2.0, 3.0]
    boundary:
      name: PolygonBoundary
      params:
        n_sides: [4, 6]
        radius: 0.2
        rotation: pi/4
    samples: 64
    """
    config = yaml.safe_load(yaml_content)
    p = PathIntegrationDatasetParams(**config)
    
    print(f"Strategy: {p.strategy}")
    print(f"Boundary: {p.boundary}")
    
    # Test with generate_param_combinations
    try:
        from project.boolean_reservoir.code.utils.param_utils import generate_param_combinations
        p_list = generate_param_combinations(p)
        
        print(f"\nGenerated {len(p_list)} combinations")
        for i, combo in enumerate(p_list[:2]):
            print(f"\n--- Combination {i+1} ---")
            print(f"  dimensions: {combo.dimensions}")
            print(f"  strategy.params: {combo.strategy.params}")
            print(f"  boundary.params: {combo.boundary.params}")
            print(f"  strategy_obj: {combo.strategy_obj}")
            print(f"  boundary_obj: {combo.boundary_obj}")
    except ImportError as e:
        print(f"\nCould not test with generate_param_combinations: {e}")
