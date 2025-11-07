from pydantic import BaseModel, Field
from typing import Tuple, Optional, List, NamedTuple

Volts = Optional[int]
Amps = Optional[float]
Watts = Optional[int]
Count = Optional[int]
Length = Optional[float]
Weight = Optional[float]

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

class BatteryInput_(BaseModel):
    v_nom: Volts = Field(description="nominal battery voltage (i.e 12V, 24V, 48V)", units="V")
    v_range: VoltRange = Field(description="supported battery voltage range", units="V")
    i_charge_ac: Amps = Field(description="maximum charge current from AC", units="V")
    i_charge_pv: Amps = Field(description="maximum charge current from PV", units="V")
    i_invert: Amps = Field(description="maximum discharge current", units="V")
    w_invert: Watts = Field(description="maximum discharge power", units="W")
BatteryInput = Optional[BatteryInput_]

class Size_(BaseModel):
    width: Length = Field(description="width of unit", units="mm")
    height: Length = Field(description="height of unit", units="mm")
    depth: Length = Field(description="depth of unit", units="mm")
Size = Optional[Size_]

class GeneralData_(BaseModel):
    size: Size = Field(description="dimentions", units="mm")
    weigth: Weight = Field(description="weight", units="kg")
    noice: Optional[float] = Field(description="maximum noice output", units="dB")
GeneralData = Optional[GeneralData_]

class HybridInverter(Commercial):
    pv_input: PVInput = Field(description="Photo-Voltaics input spec")
    battery_input: BatteryInput = Field(description="Battery parameters")
    general_data: GeneralData = Field(description="General Data")
