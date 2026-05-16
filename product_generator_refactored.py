"""
Product Listing Generator using OpenAI and Fashion Product Images Dataset.

Refactored version with:
- smaller functions with single responsibilities
- shared image encoding helper
- configurable prompt template
- explicit file and validation error handling
- user-facing messages for validation failures

TODOs for future refactoring:
- Extract configuration values into a dedicated settings object or `.env`-backed config module.
- Add a true backup model fallback instead of retrying only the same model.
- Add tests for prompt generation, retry behavior, and file-not-found paths.
- Make the retry wrapper type-safe with a `Callable[..., T]` signature.
- Parse and validate the model's JSON response before returning `output_text`.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from string import Template
from typing import Any

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

MODEL_NAME = "gpt-4o-mini"
DATASET_NAME = "ashraq/fashion-product-images-small"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_IMAGE_DIR = Path("product_images")
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 1.0
RETRYABLE_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "APIStatusError",
    "InternalServerError",
    "RateLimitError",
}

PROMPT_TEMPLATE = Template("""You are an expert e-commerce copywriter. Analyze the product image and create a compelling product listing.

Product Information:
- Name: $product_name
- Price: $$price
- Category: $category
$additional_info_line

Please create a professional product listing that includes:

1. **Product Title** (catchy, SEO-friendly, 60 characters max)
2. **Product Description** (detailed, 150-200 words)
   - Highlight key features and benefits
   - Use persuasive language
   - Include relevant details visible in the image
3. **Key Features** (bullet points, 5-7 items)
4. **SEO Keywords** (comma-separated, 10-15 relevant keywords)

Format your response as JSON with the following structure:
{{
    "title": "Product title here",
    "description": "Full description here",
    "features": ["Feature 1", "Feature 2", ...],
    "keywords": "keyword1, keyword2, ..."
}}

Be specific about what you see in the image. Mention colors, materials, design elements, and any distinctive features.""")

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for debugging and monitoring."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def load_openai_client() -> OpenAI:
    """Load environment variables and return an OpenAI client."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to your environment or .env file."
        )
    return OpenAI(api_key=api_key)


def setup_directories() -> Path:
    """Create and return the directory for local product images."""
    DEFAULT_IMAGE_DIR.mkdir(exist_ok=True)
    return DEFAULT_IMAGE_DIR


def load_dataset_from_huggingface(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Load the product dataset or fall back to local sample data."""
    # TODO: Move fallback sample data to a separate fixture or JSON file.
    logger.info("Loading product dataset...")
    try:
        dataset = load_dataset(DATASET_NAME, split=f"train[:{sample_size}]")
        products_df = pd.DataFrame(dataset)
        logger.info("Loaded %s products from Hugging Face.", len(products_df))
        return products_df
    except Exception as exc:
        logger.warning("Could not load Hugging Face dataset: %s", exc)
        logger.warning("Using local fallback sample data instead.")
        fallback_products = [
            {
                "id": 1,
                "name": "Wireless Headphones",
                "price": 79.99,
                "category": "Electronics",
                "image_path": "images/product1.jpg",
            }
        ]
        return pd.DataFrame(fallback_products)


def validate_products_dataframe(products_df: pd.DataFrame) -> None:
    """Validate that the dataframe has enough data to continue."""
    if products_df.empty:
        raise ValueError("No products were loaded. The dataset is empty.")

    if "image" not in products_df.columns and "image_path" not in products_df.columns:
        raise ValueError(
            "The dataset must include either an 'image' column or an 'image_path' column."
        )


def get_image_value(product_row: pd.Series) -> Any:
    """Return the best available image value from a dataset row."""
    if "image" in product_row and pd.notna(product_row["image"]):
        return product_row["image"]
    if "image_path" in product_row and pd.notna(product_row["image_path"]):
        return product_row["image_path"]
    raise ValueError("The selected product does not include an image or image path.")


def build_product_listing_prompt(
    product_name: str,
    price: float,
    category: str,
    additional_info: str | None = None,
) -> str:
    """Build a prompt using a template instead of hardcoding the text inline."""
    additional_info_line = (
        f"- Additional Info: {additional_info}" if additional_info else ""
    )
    return PROMPT_TEMPLATE.substitute(
        product_name=product_name,
        price=f"{price:.2f}",
        category=category,
        additional_info_line=additional_info_line,
    )


def encode_image_to_base64(image_value: Any) -> str:
    """
    Encode either a PIL image or a file path to base64.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
    """
    if isinstance(image_value, Image.Image) or hasattr(image_value, "save"):
        buffer = BytesIO()
        image_value.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    image_path = Path(str(image_value))
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def is_retryable_exception(exc: Exception) -> bool:
    """Return True when the exception looks transient and worth retrying."""
    return exc.__class__.__name__ in RETRYABLE_ERROR_NAMES


def execute_with_retry(
    operation,
    *,
    operation_name: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
):
    """
    Execute a callable with exponential backoff for retryable failures.

    This wrapper is intentionally reusable so any future OpenAI request can
    benefit from the same retry/error behavior.
    """
    # TODO: Add precise typing with Callable and a generic return type.
    attempt = 0
    while True:
        try:
            logger.debug("Running %s (attempt %s).", operation_name, attempt + 1)
            return operation()
        except FileNotFoundError:
            raise
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not is_retryable_exception(exc):
                logger.exception(
                    "%s failed after %s attempt(s).", operation_name, attempt
                )
                raise RuntimeError(
                    f"{operation_name} failed after {attempt} attempt(s): {exc}"
                ) from exc

            delay = base_delay_seconds * (2 ** (attempt - 1))
            logger.warning(
                "%s failed with a transient error (%s). Retrying in %.1fs... (%s/%s)",
                operation_name,
                exc.__class__.__name__,
                delay,
                attempt,
                max_retries,
            )
            time.sleep(delay)


def call_openai(
    client: OpenAI, image_value: Any, text_prompt: str, model: str = MODEL_NAME
):
    """Call OpenAI with image and text prompt through a reusable retry wrapper."""
    # TODO: Add a model fallback sequence if the primary model is unavailable.
    logger.debug("Preparing OpenAI request using model %s.", model)
    base64_image = encode_image_to_base64(image_value)

    def request() -> Any:
        return client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text_prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                }
            ],
        )

    return execute_with_retry(request, operation_name="OpenAI request")


def generate_first_product_listing() -> str:
    """Generate a listing for the first valid product in the dataset."""
    client = load_openai_client()
    setup_directories()

    products_df = load_dataset_from_huggingface()
    validate_products_dataframe(products_df)

    logger.info("Dataset prepared.")
    logger.info("Total products: %s", len(products_df))

    first_product = products_df.iloc[0]
    image_value = get_image_value(first_product)

    product_name = str(first_product.get("name", "Unknown Product"))
    price = float(first_product.get("price", 0.0) or 0.0)
    category = str(first_product.get("category", "Uncategorized"))
    additional_info = first_product.get("description") or first_product.get("details")

    prompt = build_product_listing_prompt(
        product_name=product_name,
        price=price,
        category=category,
        additional_info=str(additional_info) if additional_info else None,
    )

    response = call_openai(client, image_value, prompt)
    return response.output_text


def main() -> None:
    """Main execution function with explicit user-facing errors."""
    setup_logging()
    try:
        logger.info("Starting product listing generation.")
        output_text = generate_first_product_listing()
        logger.info("Generated product listing successfully.")
        print("\nGenerated Product Listing:")
        print(output_text)
    except FileNotFoundError as exc:
        logger.error("File error: %s", exc)
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)


if __name__ == "__main__":
    main()
