import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_FILE = "retraction_watch.csv"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

df = df[df["RetractionNature"].str.lower().eq("retraction")].copy()

paper_mill = df[
    df["Reason"].fillna("").str.contains("paper mill", case=False, regex=False)
].copy()

paper_mill["author_count"] = paper_mill["Author"].fillna("").apply(
    lambda x: len([a for a in x.split(";") if a.strip()])
)

paper_mill = paper_mill[paper_mill["author_count"] > 0].copy()

invalid_authors = ["unknown", "anonymous", "anon"]

paper_mill = paper_mill[
    ~paper_mill["Author"]
    .fillna("")
    .str.strip()
    .str.lower()
    .isin(invalid_authors)
].copy()

paper_mill["authorship_type"] = paper_mill["author_count"].apply(
    lambda n: "solo" if n == 1 else "multi"
)

paper_mill["year"] = pd.to_datetime(
    paper_mill["OriginalPaperDate"],
    errors="coerce"
).dt.year

paper_mill = paper_mill[
    paper_mill["year"].between(2000, 2024)
].copy()

bins = [2000, 2004, 2008, 2012, 2016, 2020, 2025]
labels = ["2000-2003", "2004-2007", "2008-2011", "2012-2015", "2016-2019", "2020-2024"]

paper_mill["period"] = pd.cut(
    paper_mill["year"],
    bins=bins,
    labels=labels,
    right=False
)

paper_mill["publication_type"] = "journal"

conference_keywords = ["conference", "proceedings", "symposium", "workshop"]

mask = paper_mill["Journal"].str.lower().str.contains(
    "|".join(conference_keywords),
    na=False
)

paper_mill.loc[mask, "publication_type"] = "conference"

summary = paper_mill["authorship_type"].value_counts().rename_axis("authorship_type").reset_index(name="count")
summary["percentage"] = (summary["count"] / summary["count"].sum() * 100).round(1)
summary.to_csv(OUTPUT_DIR / "summary_authorship_type.csv", index=False)

trend = pd.crosstab(
    paper_mill["period"],
    paper_mill["authorship_type"],
    normalize="index"
) * 100

trend.round(1).to_csv(OUTPUT_DIR / "table1_temporal_dynamics.csv")

solo = paper_mill[paper_mill["authorship_type"] == "solo"].copy()

publisher_table = solo["Publisher"].value_counts().reset_index()
publisher_table.columns = ["Publisher", "Number_of_retractions"]
publisher_table["Percentage"] = (
    publisher_table["Number_of_retractions"] /
    publisher_table["Number_of_retractions"].sum() * 100
).round(1)

top6 = publisher_table.head(6).copy()
other = pd.DataFrame({
    "Publisher": ["Other publishers"],
    "Number_of_retractions": [publisher_table.iloc[6:]["Number_of_retractions"].sum()],
    "Percentage": [round(publisher_table.iloc[6:]["Percentage"].sum(), 1)]
})

table2 = pd.concat([top6, other], ignore_index=True)
table2.to_csv(OUTPUT_DIR / "table2_top_publishers_solo.csv", index=False)

paper_mill["author_count_grouped"] = paper_mill["author_count"].apply(
    lambda x: str(x) if x <= 10 else "11+"
)

order = [str(i) for i in range(1, 11)] + ["11+"]

author_dist = (
    paper_mill["author_count_grouped"]
    .value_counts()
    .reindex(order)
)

plt.figure(figsize=(8, 5))
plt.bar(author_dist.index, author_dist.values)
plt.xlabel("Number of authors")
plt.ylabel("Number of retractions")
plt.title("Distribution of author counts in paper mill-related retractions")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure1_author_count_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

pub_type = pd.crosstab(
    paper_mill["publication_type"],
    paper_mill["authorship_type"],
    normalize="index"
) * 100

pub_type = pub_type.loc[["conference", "journal"], ["solo", "multi"]]

ax = pub_type.plot(
    kind="bar",
    stacked=True,
    figsize=(7, 5)
)

plt.ylabel("Percentage of retractions")
plt.xlabel("Publication type")
plt.title("Authorship patterns by publication type")
ax.legend(
    ["Solo-authored", "Multi-authored"],
    title="Authorship type",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2
)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure2_conference_vs_journal.png", dpi=300, bbox_inches="tight")
plt.close()

paper_mill.to_csv(OUTPUT_DIR / "paper_mill_retractions_processed.csv", index=False)

print("Analysis completed.")
print(f"Paper mill retractions: {len(paper_mill)}")
print(summary)
