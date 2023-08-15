"""Python script to run the notebook"""

import sys
from config import Location
from prefect import flow
from prefect_jupyter import notebook

from architectures import get_architecture


@flow
def run_notebook(location: Location, params):
    """Run a notebook with specified parameters then
    generate a notebook with the outputs

    Parameters
    ----------
    location : Location, optional
        Locations of inputs and outputs, by default Location()
    """

    if not params:
        params = {
            "test_data_location": location.test_data,
            "predictions_location": location.predictions,
        }

    nb = notebook.execute_notebook(
        location.results_template_notebook,
        parameters=params,
    )
    body = notebook.export_notebook(nb)
    with open(location.results_notebook, "w") as f:
        f.write(body)


if __name__ == "__main__":
    config = get_architecture(sys.argv[1])
    run_notebook(location=config["location"])
