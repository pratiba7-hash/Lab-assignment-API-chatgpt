"""Orchestration entrypoint for the product listing generator."""

from __future__ import annotations

from helpers import (
    build_product_listing_prompt,
    call_openai,
    get_image_value,
    load_dataset_from_huggingface,
    load_openai_client,
    logger,
    setup_directories,
    setup_logging,
    validate_products_dataframe,
)


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
