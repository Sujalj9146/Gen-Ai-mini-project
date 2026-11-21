"""Pandas aggregation of tagged questions into a topic frequency table."""

from __future__ import annotations

import pandas as pd


def to_dataframe(tagged: list) -> pd.DataFrame:
    """Build the working DataFrame from tagged questions."""
    rows = []
    for item in tagged:
        rows.append(
            {
                "Question ID": item.question_id,
                "Paper": item.source_file,
                "Q No": item.number,
                "Marks": item.marks,
                "Topic": item.topic,
                "Question": " ".join(item.text.split()),
            }
        )
    return pd.DataFrame(rows)


def topic_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many times each topic was asked, most frequent first.

    Also reports the share of questions and the total marks attached to each
    topic, because a topic worth 15 marks across three questions matters more
    than one worth 6 marks across a single 2-mark question.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["Topic", "Questions", "% of Questions", "Total Marks"]
        )

    grouped = (
        df.groupby("Topic", dropna=False)
        .agg(
            Questions=("Question ID", "count"),
            Total_Marks=("Marks", "sum"),
            Papers=("Paper", "nunique"),
        )
        .reset_index()
    )

    total_questions = len(df)
    grouped["% of Questions"] = (grouped["Questions"] / total_questions * 100).round(1)

    grouped = grouped.rename(columns={"Total_Marks": "Total Marks", "Papers": "Papers Covered"})
    grouped = grouped.sort_values(
        by=["Questions", "Total Marks"], ascending=[False, False]
    ).reset_index(drop=True)

    ordered = ["Topic", "Questions", "% of Questions", "Total Marks", "Papers Covered"]
    return grouped[ordered]


def papers_covered(df: pd.DataFrame) -> pd.DataFrame:
    """Per-paper question and topic counts."""
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("Paper")
        .agg(Questions=("Question ID", "count"), Topics=("Topic", "nunique"))
        .reset_index()
        .sort_values("Questions", ascending=False)
    )


def topic_by_paper_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Topics down the side, papers across the top, counts inside."""
    if df.empty:
        return pd.DataFrame()
    matrix = pd.crosstab(df["Topic"], df["Paper"])
    return matrix.sort_values(by=list(matrix.columns), ascending=False)


def chart_data(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Top-N topics ready for a bar chart."""
    counts = topic_counts(df)
    if counts.empty:
        return counts
    return counts.head(top_n)
