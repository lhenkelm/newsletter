import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    df = pl.read_parquet("hf://datasets/zenml/llmops-database/data/train-00000-of-00001.parquet")
    df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
