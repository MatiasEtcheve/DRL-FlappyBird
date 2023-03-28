from typing import Tuple, List

BarObs = Tuple[float, float, float, bool]
BirdObs = Tuple[float, float, float]
FlappyObs = Tuple[BirdObs, List[BarObs]]