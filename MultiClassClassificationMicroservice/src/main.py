from typing import (
    Dict,
    List
)
from fastapi import (
    FastAPI,
    Depends
)

from src.constants import MAX_OBJECTS_BATCH
from src.core.processing_factory import ProcessorFactory
from src.utils.sentry import configure_sentry
from src.utils.versioning import get_version
from src.utils.logger import (
    configure_logger,
    handle_exception
)


# Initialize factory before FastAPI starts(when pads starts)
processor_factory = ProcessorFactory.get_processor()
# Initialize FastAPI application
app = FastAPI(title="Multi-classification service")
# Configure Sentry for error tracking
configure_sentry()
# Configure logger for logging service events and errors
logger = configure_logger()
# Load service version from versioning module
VERSION = get_version()


@app.get("/health-check/")
async def health_check():
    """
    Endpoint to check the health status of the service.
    Returns:
        dict: A dictionary with service name and version tag.
    """
    return {"service": "co_pr_service", "tag": VERSION}


@app.post(f"/co_pr_service/predictions/", tags=["co_pr_service"])
async def analyze(
        data: List[Dict],
        max_obj_n: int = MAX_OBJECTS_BATCH,
        _: TokenValidator = Depends(get_token_validator)) -> dict:
    """
    Endpoint to process data and return predictions based on private or corporative flow.
    Args:
        data (List[Dict]): List of data dictionaries to process.
        max_obj_n (int): Maximum number of objects to process in a batch. Default is 6.
        _: TokenValidator: Dependency to validate the token.
    Returns:
        dict: The result of the processing.
    """
    try:
        # Execute the country-specific processor with provided data and batch size
        result = processor_factory.execute(data, max_obj_n)
        return result

    except Exception as e:
        # Handle exceptions and log errors
        handle_exception(e, logger)
