from pydantic import BaseModel
import sensorkit.api as sk


class SKOffer(BaseModel):
    pass


class SKCollectTaskTLE(sk.CollectTask):
    pass


class SKCollectTaskRADec(sk.CollectTask):
    ra_deg: float
    dec_deg: float


class SKCollectTaskCustomRate(sk.CollectTask):
    ra_deg: float
    dec_deg: float
    ra_rate_arcsec_per_second: float
    dec_rate_arcsec_per_second: float


class SKCollectTaskSV(sk.CollectTask):
    pass
