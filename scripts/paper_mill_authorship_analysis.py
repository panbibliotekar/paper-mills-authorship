import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# =============================================================================
# Configuration
# =============================================================================

START_YEAR = 2000
END_YEAR = 2024
EXPECTED_FINAL_N = 11704

OUTPUT_DIR = Path("outputs")

REQUIRED_COLUMNS = {
    "RetractionNature",
    "Reason",
    "Author",
    "OriginalPaperDate",
    "Journal",
    "Publisher",
    "ArticleType",
}

# =============================================================================
# Helper functions
# =============================================================================

def load_data(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path, low_memory=False)

    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path)

    else:
        raise ValueError(
            "Unsupported input format. Please use CSV, XLSX, or XLS."
        )

    # Remove empty Excel-generated columns such as "Unnamed: 20"
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

    return df

def validate_columns(df):
    
    missing = REQUIRED_COLUMNS.difference(df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

def count_authors(author_string):

    if pd.isna(author_string):
        return 0

    authors = [
        author.strip()
        for author in str(author_string).split(";")
        if author.strip()
    ]

    return len(authors)

# =============================================================================
# Data cleaning
# =============================================================================

def clean_data(df):

    cleaning_log = []

    # -------------------------------------------------------------------------
    # Initial dataset
    # -------------------------------------------------------------------------

    cleaning_log.append({
        "stage": "Input records",
        "n": len(df)
    })

    # -------------------------------------------------------------------------
    # Retractions only
    # -------------------------------------------------------------------------

    df = df[
        df["RetractionNature"]
        .fillna("")
        .str.strip()
        .str.casefold()
        .eq("retraction")
    ].copy()

    cleaning_log.append({
        "stage": "RetractionNature = Retraction",
        "n": len(df)
    })

    # -------------------------------------------------------------------------
    # Paper-mill-related records
    # -------------------------------------------------------------------------

    df = df[
        df["Reason"]
        .fillna("")
        .str.contains(
            "paper mill",
            case=False,
            regex=False
        )
    ].copy()

    cleaning_log.append({
        "stage": "Reason contains 'paper mill'",
        "n": len(df)
    })

    # -------------------------------------------------------------------------
    # Parse publication year
    # -------------------------------------------------------------------------

    df["year"] = pd.to_datetime(
        df["OriginalPaperDate"],
        errors="coerce"
    ).dt.year

    invalid_dates = df[df["year"].isna()].copy()

    if not invalid_dates.empty:
        invalid_dates.to_csv(
            OUTPUT_DIR / "excluded_invalid_publication_dates.csv",
            index=False
        )

    df = df[df["year"].notna()].copy()

    cleaning_log.append({
        "stage": "Valid OriginalPaperDate",
        "n": len(df)
    })

    # -------------------------------------------------------------------------
    # Publication window
    # -------------------------------------------------------------------------

    outside_window = df[
        ~df["year"].between(
            START_YEAR,
            END_YEAR
        )
    ].copy()

    if not outside_window.empty:
        outside_window.to_csv(
            OUTPUT_DIR / "excluded_outside_year_window.csv",
            index=False
        )

    df = df[
        df["year"].between(
            START_YEAR,
            END_YEAR
        )
    ].copy()

    cleaning_log.append({
        "stage": f"Original publication year {START_YEAR}-{END_YEAR}",
        "n": len(df)
    })

    # -------------------------------------------------------------------------
    # Exclude records without identifiable authors
    # -------------------------------------------------------------------------

    normalized_author = (
        df["Author"]
        .fillna("")
        .str.strip()
        .str.casefold()
    )

    invalid_author_mask = normalized_author.isin(
        ["", "unknown"]
    )

    invalid_authors = df[
        invalid_author_mask
    ].copy()

    if not invalid_authors.empty:
        invalid_authors.to_csv(
            OUTPUT_DIR / "excluded_invalid_authors.csv",
            index=False
        )

    df = df[
        ~invalid_author_mask
    ].copy()

    cleaning_log.append({
        "stage": "Identifiable author information",
        "n": len(df)
    })

    # -------------------------------------------------------------------------
    # Author counts
    # -------------------------------------------------------------------------

    df["author_count"] = (
        df["Author"]
        .apply(count_authors)
    )

    zero_authors = df[
        df["author_count"] == 0
    ].copy()

    if not zero_authors.empty:
        zero_authors.to_csv(
            OUTPUT_DIR / "excluded_zero_author_count.csv",
            index=False
        )

        df = df[
            df["author_count"] > 0
        ].copy()

    cleaning_log.append({
        "stage": "At least one countable author",
        "n": len(df)
    })

    # -------------------------------------------------------------------------
    # Duplicate Record ID check
    # -------------------------------------------------------------------------

    if "Record ID" in df.columns:

        duplicate_mask = df["Record ID"].duplicated(
            keep=False
        )

        duplicates = df[
            duplicate_mask
        ].copy()

        if not duplicates.empty:

            duplicates.to_csv(
                OUTPUT_DIR / "duplicate_record_ids.csv",
                index=False
            )

            raise ValueError(
                "Duplicate Record ID values detected. "
                "See outputs/duplicate_record_ids.csv."
            )

    # -------------------------------------------------------------------------
    # Final sample-size validation
    # -------------------------------------------------------------------------

    final_n = len(df)

    cleaning_log.append({
        "stage": "Final analytic dataset",
        "n": final_n
    })

    cleaning_log = pd.DataFrame(
        cleaning_log
    )

    cleaning_log.to_csv(
        OUTPUT_DIR / "cleaning_summary.csv",
        index=False
    )

    if final_n != EXPECTED_FINAL_N:

        raise ValueError(
            f"Expected final analytic N = "
            f"{EXPECTED_FINAL_N:,}, "
            f"but obtained {final_n:,}. "
            f"Check the input data and "
            f"outputs/cleaning_summary.csv."
        )

    # -------------------------------------------------------------------------
    # Authorship classification
    # -------------------------------------------------------------------------

    df["authorship_type"] = df[
        "author_count"
    ].apply(
        lambda n: "solo"
        if n == 1
        else "multi"
    )

    return df, cleaning_log

# =============================================================================
# Temporal grouping
# =============================================================================

def add_time_periods(df):

    bins = [
        2000,
        2004,
        2008,
        2012,
        2016,
        2020,
        2025
    ]

    labels = [
        "2000-2003",
        "2004-2007",
        "2008-2011",
        "2012-2015",
        "2016-2019",
        "2020-2024"
    ]

    df["period"] = pd.cut(
        df["year"],
        bins=bins,
        labels=labels,
        right=False
    )

    return df

# =============================================================================
# Publication-type classification
# =============================================================================

def classify_publication_type(df):

    df["publication_type"] = "non-conference"

    conference_mask = (
        df["ArticleType"]
        .fillna("")
        .str.contains(
            "Conference Abstract/Paper",
            case=False,
            regex=False
        )
    )

    df.loc[
        conference_mask,
        "publication_type"
    ] = "conference"

    return df


# =============================================================================
# Summary statistics
# =============================================================================

def create_authorship_summary(df):

    summary = (
        df["authorship_type"]
        .value_counts()
        .reindex(["solo", "multi"])
        .rename_axis("authorship_type")
        .reset_index(name="count")
    )

    summary["percentage"] = (
        summary["count"]
        / len(df)
        * 100
    ).round(1)

    summary.to_csv(
        OUTPUT_DIR / "summary_authorship_type.csv",
        index=False
    )

    descriptive = pd.DataFrame({
        "metric": [
            "Total records",
            "Mean authors",
            "Median authors",
            "Minimum authors",
            "Maximum authors"
        ],
        "value": [
            len(df),
            round(df["author_count"].mean(), 2),
            df["author_count"].median(),
            df["author_count"].min(),
            df["author_count"].max()
        ]
    })

    descriptive.to_csv(
        OUTPUT_DIR / "summary_author_counts.csv",
        index=False
    )

    return summary

# =============================================================================
# Author-count distribution
# =============================================================================

def create_author_distribution(df):

    distribution = (
        df["author_count"]
        .value_counts()
        .sort_index()
        .rename_axis("author_count")
        .reset_index(name="count")
    )

    distribution["percentage"] = (
        distribution["count"]
        / len(df)
        * 100
    ).round(2)

    distribution.to_csv(
        OUTPUT_DIR / "author_count_distribution.csv",
        index=False
    )

    return distribution

# =============================================================================
# Temporal analysis
# =============================================================================

def create_temporal_table(df):

    counts = pd.crosstab(
        df["period"],
        df["authorship_type"]
    )

    counts = counts.reindex(
        columns=["solo", "multi"],
        fill_value=0
    )

    counts["total"] = (
        counts["solo"]
        + counts["multi"]
    )

    counts["solo_percentage"] = (
        counts["solo"]
        / counts["total"]
        * 100
    ).round(1)

    counts["multi_percentage"] = (
        counts["multi"]
        / counts["total"]
        * 100
    ).round(1)

    counts.to_csv(
        OUTPUT_DIR / "table1_temporal_dynamics.csv"
    )

    return counts

# =============================================================================
# Publisher analysis
# =============================================================================

def create_publisher_table(df):

    solo_recent = df[
        (df["authorship_type"] == "solo")
        & (df["year"].between(2020, 2024))
    ].copy()

    publisher_table = (
        solo_recent["Publisher"]
        .value_counts()
        .rename_axis("Publisher")
        .reset_index(name="Number_of_retractions")
    )

    publisher_table["Percentage"] = (
        publisher_table[
            "Number_of_retractions"
        ]
        / len(solo_recent)
        * 100
    ).round(1)

    top6 = publisher_table.head(6).copy()

    other_count = (
        publisher_table
        .iloc[6:]["Number_of_retractions"]
        .sum()
    )

    other_percentage = (
        other_count
        / len(solo_recent)
        * 100
    )

    other = pd.DataFrame({
        "Publisher": [
            "Other publishers"
        ],
        "Number_of_retractions": [
            other_count
        ],
        "Percentage": [
            round(
                other_percentage,
                1
            )
        ]
    })

    table2 = pd.concat(
        [top6, other],
        ignore_index=True
    )

    table2.to_csv(
        OUTPUT_DIR /
        "table2_top_publishers_solo_2020_2024.csv",
        index=False
    )

    return table2


# =============================================================================
# Publication-type table
# =============================================================================

def create_publication_type_table(df):

    counts = pd.crosstab(
        df["publication_type"],
        df["authorship_type"]
    )

    counts = counts.reindex(
        index=[
            "conference",
            "non-conference"
        ],
        columns=[
            "solo",
            "multi"
        ],
        fill_value=0
    )

    counts["total"] = (
        counts["solo"]
        + counts["multi"]
    )

    counts["solo_percentage"] = (
        counts["solo"]
        / counts["total"]
        * 100
    ).round(1)

    counts["multi_percentage"] = (
        counts["multi"]
        / counts["total"]
        * 100
    ).round(1)

    counts.to_csv(
        OUTPUT_DIR /
        "publication_type_authorship.csv"
    )

    return counts

# =============================================================================
# Figure 1
# =============================================================================

def create_figure1(df):

    df["author_count_grouped"] = (
        df["author_count"]
        .apply(
            lambda x:
            str(x)
            if x <= 10
            else "11+"
        )
    )

    order = (
        [str(i) for i in range(1, 11)]
        + ["11+"]
    )

    author_dist = (
        df["author_count_grouped"]
        .value_counts()
        .reindex(order, fill_value=0)
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        author_dist.index,
        author_dist.values
    )

    plt.xlabel(
        "Number of authors"
    )

    plt.ylabel(
        "Number of retractions"
    )

    plt.title(
        "Distribution of author counts "
        "in paper-mill-related retractions"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "figure1_author_count_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =============================================================================
# Figure 2
# =============================================================================

def create_figure2(df):

    percentages = pd.crosstab(
        df["publication_type"],
        df["authorship_type"],
        normalize="index"
    ) * 100

    percentages = percentages.reindex(
        index=[
            "conference",
            "non-conference"
        ],
        columns=[
            "solo",
            "multi"
        ],
        fill_value=0
    )

    ax = percentages.plot(
        kind="bar",
        stacked=True,
        figsize=(7, 5)
    )

    plt.ylabel(
        "Percentage of retractions"
    )

    plt.xlabel(
        "Publication type"
    )

    plt.title(
        "Authorship patterns by publication type"
    )

    ax.legend(
        [
            "Solo-authored",
            "Multi-authored"
        ],
        title="Authorship type",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "figure2_publication_type.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =============================================================================
# Main analysis
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyse authorship patterns "
            "among paper-mill-related "
            "retractions in the "
            "Retraction Watch Database."
        )
    )

    parser.add_argument(
        "data_file",
        nargs="?",
        default="retraction_watch.csv",
        help=(
            "Path to the Retraction Watch "
            "CSV or Excel file "
            "(default: retraction_watch.csv)"
        )
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    print(
        f"Loading data from: "
        f"{args.data_file}"
    )

    df = load_data(
        args.data_file
    )

    validate_columns(
        df
    )

    paper_mill, cleaning_log = (
        clean_data(df)
    )

    paper_mill = (
        add_time_periods(
            paper_mill
        )
    )

    paper_mill = (
        classify_publication_type(
            paper_mill
        )
    )

    summary = (
        create_authorship_summary(
            paper_mill
        )
    )

    create_author_distribution(
        paper_mill
    )

    temporal_table = (
        create_temporal_table(
            paper_mill
        )
    )

    publisher_table = (
        create_publisher_table(
            paper_mill
        )
    )

    publication_type_table = (
        create_publication_type_table(
            paper_mill
        )
    )

    create_figure1(
        paper_mill
    )

    create_figure2(
        paper_mill
    )

    # Save definitive processed analytic dataset
    paper_mill.to_csv(
        OUTPUT_DIR /
        "paper_mill_retractions_processed.csv",
        index=False
    )

    print()
    print(
        "Analysis completed successfully."
    )

    print()
    print(
        "Cleaning summary:"
    )

    print(
        cleaning_log.to_string(
            index=False
        )
    )

    print()
    print(
        f"Final analytic N: "
        f"{len(paper_mill):,}"
    )

    print()
    print(
        "Authorship summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Temporal dynamics:"
    )

    print(
        temporal_table.to_string()
    )

    print()
    print(
        "Publication-type analysis:"
    )

    print(
        publication_type_table.to_string()
    )

    print()
    print(
        "Top publishers among "
        "solo-authored records, 2020-2024:"
    )

    print(
        publisher_table.to_string(
            index=False
        )
    )

if __name__ == "__main__":
    main()
