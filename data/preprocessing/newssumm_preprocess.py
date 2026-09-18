import difflib
import os
import re
import pandas as pd



# 1. Define the 36 valid canonical newspaper names
CANONICAL_NEWSPAPERS = [
    "The Times of India",
    "The Hindu",
    "Hindustan Times",
    "Indian Express",
    "The Economic Times",
    "Business Standard",
    "The Mint",
    "Financial Express",
    "Deccan Chronicle",
    "The Telegraph",
    "Deccan Herald",
    "The Pioneer",
    "The Statesman",
    "The Tribune",
    "Mid-Day",
    "Mumbai Mirror",
    "Pune Mirror",
    "Bangalore Mirror",
    "Ahmedabad Mirror",
    "DNA",
    "Firstpost",
    "Free Press Journal",
    "Navhind Times",
    "Sentinel Assam",
    "The Asian Age",
    "The Shillong Times",
    "Imphal Free Press",
    "Orissa POST",
    "Hitavada",
    "Nagaland Post",
    "Sikkim Express",
    "Greater Kashmir",
    "Kashmir Observer",
    "Daily Excelsior",
    "The Millennium Post",
    "Central Chronicle"
]

# Create a normalized lowercase lookup dictionary
canonical_lookup = {
    name.lower().strip(): name for name in CANONICAL_NEWSPAPERS
}


def clean_newspaper_name(val, cutoff=0.45):
    """Cleans garbage entries and fuzzy-matches typos to canonical names."""
    if pd.isna(val):
        return None

    text = str(val).lower().strip()

    # Step A: Filter out non-names (URLs, scraper instructions)
    if (
        text.startswith("http")
        or "please write" in text
        or "newspaper must be" in text
    ):
        return None

    # Step B: If multiple names are joined by slash/comma, take the primary one
    text = re.split(r"[/,]", text)[0].strip()

    # Step C: Check for exact normalized match
    if text in canonical_lookup:
        return canonical_lookup[text]

    # Step D: Fuzzy match against canonical names for typos (e.g., 'tbe times of india')
    matches = difflib.get_close_matches(
        text, canonical_lookup.keys(), n=1, cutoff=cutoff
    )
    if matches:
        return canonical_lookup[matches[0]]

    # Return None if it cannot be matched to one of the 36 valid names
    return None


# --- Main Pipeline Execution ---
file_path = os.path.join(
    "data", "raw", "NewsSumm", "RawDateset", "NewsSumm.xlsx"
)
df = pd.read_excel(file_path, engine="openpyxl")

# Apply cleaning
df["newspaper_name"] = df["newspaper_name"].apply(clean_newspaper_name)


# Inspection
print("--- Value Counts After Cleaning ---")
print(df["newspaper_name"].value_counts(dropna=False))
print(f"\nTotal Unique Clean Names: {df['newspaper_name'].nunique()}")

# Dropping rows where the cleaned newspaper name is None (unmatched or invalid)
df = df.dropna(subset=["newspaper_name"])

# Saving the processed dataset to a XLSX file

output_dir = os.path.join(
    "data", "raw", "NewsSumm", "Processed"
)
output_file = os.path.join(output_dir, "NewsSumm_processed.xlsx")

print(f"\nSaving cleaned dataset to: {output_file}")

df.to_excel(output_file, index=False, engine="openpyxl")

print("Done! Processed dataset successfully created.")

