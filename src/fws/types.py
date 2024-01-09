from typing import Annotated, Literal, TypeVar
import numpy as np
import numpy.typing as npt

# see: https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype

DType = TypeVar("DType", bound=np.generic)

ArrayNxNx2 = Annotated[npt.NDArray[DType], Literal["N", "N", 2]]