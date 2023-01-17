from pydantic import BaseModel, Field
from typing import Tuple, Optional, List, NamedTuple

Volts = Optional[int]
Amps = Optional[float]
Watts = Optional[int]
Count = Optional[int]

class Commercial(BaseModel):
    brand: Optional[str]
    model: Optional[str]
    tags: Optional[List[str]]

# VoltRange_ = NamedTuple('VoltRange', [('min', Volts), ('max', Volts)])
# VoltRange = Optional[VoltRange_]
class VoltRange_(BaseModel):
    min: Volts
    max: Volts
VoltRange = Optional[VoltRange_]

class PVInput_(BaseModel):
    v_nom: Volts = Field(description="nominal voltage (best efficiency)", units="V")
    v_crit: VoltRange = Field(description="allowed voltage range", units="V")
    v_mppt: VoltRange = Field(description="mppt voltage range", units="V")
    v_start: VoltRange = Field(description="off->on transition range points", units="V")
    w_max: Watts = Field(description="maximum input power", units="W")
    i_nom: Optional[List[Amps]] = Field(description="nominal current (per string)", units="A")
    i_sc: Optional[List[Amps]] = Field(description="maximum short circuit current (per string)", units="A")
    n_mppt: Count = Field(description="number of MPPT trackers", units="n")
    n_str: Optional[List[Count]] = Field(description="number of input strings per tracker", units="n")
PVInput = Optional[PVInput_]

class HybridInverter(Commercial):
    pv_input: PVInput = Field(description="Photo-Voltaics input spec")
