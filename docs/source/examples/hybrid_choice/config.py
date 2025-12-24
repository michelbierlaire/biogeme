from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Config:
    name: str
    latent_variables: Literal["zero", "two"]
    choice_model: Literal["yes", "no"]
    estimation: Literal["bayes", "ml"]
    number_of_bayesian_draws_per_chain: int
    number_of_monte_carlo_draws: int
