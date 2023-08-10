from architectures.centralized.model import CentralizedModel
from architectures.federated.model import FedratedModel
from architectures.split.model import SplitModel
from config import CentralizedLocation, FederatedLocation, SplitLocation


def get_architecture(id: str):
    """Get Requested Architecture"""
    match id:
        case "CENTRALIZED":
            return {
                "model": CentralizedModel(),
                "location": CentralizedLocation,
            }
        case "FEDERATED":
            return {"model": FedratedModel(), "location": FederatedLocation}
        case "SPLIT":
            return {"model": SplitModel(), "location": SplitLocation}
        case _:
            raise ValueError(f"Invalid architecture id: {id}")
