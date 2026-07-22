from agents.steps import (
    PIPELINE,
    PreprocessingAgent,
    StructureAgent,
    MissingValuesAgent,
    OutlierAgent,
    CorrelationAgent,
    DistributionAgent,
    AssumptionsAgent,
    DimensionalityAgent,
    MultivariateAgent,
    SummaryAgent,
)

AGENT_REGISTRY = {cls().name: cls for cls in PIPELINE}

__all__ = [
    "PIPELINE", "AGENT_REGISTRY",
    "PreprocessingAgent", "StructureAgent", "MissingValuesAgent", "OutlierAgent",
    "CorrelationAgent", "DistributionAgent", "AssumptionsAgent", "DimensionalityAgent",
    "MultivariateAgent", "SummaryAgent",
]
