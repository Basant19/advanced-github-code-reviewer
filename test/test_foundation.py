import sys
from app.core.logger import get_logger
from app.core.exceptions import CustomException

logger = get_logger(__name__)

def test_exception_flow():
    try:
        logger.info("Starting foundation test...")
        # Intentional error: Division by zero
        result = 1 / 0
    except Exception as e:
        logger.error("A division error occurred!")
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        test_exception_flow()
    except CustomException as ce:
        print("\n--- TEST SUCCESSFUL ---")
        print(ce)
        logger.info(f"Captured Custom Error: {ce}")