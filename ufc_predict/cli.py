"""Command-line interface for UFC fight prediction."""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path

import click
import pandas as pd

from .config import get_config
from .pipeline import UFCPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", default=None, help="Path to config YAML")
@click.option("--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, config, verbose):
    """UFC Fight Prediction System."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option("--source", type=click.Choice(["ufcstats", "sherdog", "odds", "all"]),
              default="all", help="Data source to scrape")
@click.option("--output", default="data/raw", help="Output directory")
@click.pass_context
def scrape(ctx, source, output):
    """Scrape fight data from configured sources."""
    from .scrapers import UFCStatsScraper, SherdogScraper, OddsScraper

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source in ("ufcstats", "all"):
        click.echo("Scraping UFC Stats...")
        scraper = UFCStatsScraper(cache_dir="data/cache")
        df = scraper.scrape_all()
        df.to_csv(output_dir / "ufcstats_fights.csv", index=False)
        click.echo(f"  Saved {len(df)} fights to ufcstats_fights.csv")

    if source in ("odds", "all"):
        click.echo("Scraping BestFightOdds...")
        scraper = OddsScraper(cache_dir="data/cache")
        df = scraper.scrape_all()
        df.to_csv(output_dir / "odds_data.csv", index=False)
        click.echo(f"  Saved {len(df)} fight odds to odds_data.csv")

    click.echo("Done!")


@cli.command()
@click.option("--data", required=True, help="Path to fight data CSV/parquet")
@click.option("--output", default="data/processed/features.parquet", help="Output path for feature matrix")
@click.pass_context
def build_features(ctx, data, output):
    """Build feature matrix from fight data."""
    config = get_config(ctx.obj["config_path"])
    pipeline = UFCPipeline(ctx.obj["config_path"])

    click.echo(f"Loading data from {data}...")
    fights_df = pipeline.load_data(data)
    click.echo(f"  {len(fights_df)} fights loaded")

    click.echo("Building features (chronological processing)...")
    feature_df = pipeline.prepare_features(fights_df)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(output, index=False)
    click.echo(f"Feature matrix saved: {feature_df.shape[0]} fights x {feature_df.shape[1]} features")
    click.echo(f"  Output: {output}")


@cli.command()
@click.option("--features", required=True, help="Path to feature matrix")
@click.option("--save-model", default="models/pipeline.pkl", help="Path to save trained model")
@click.pass_context
def train(ctx, features, save_model):
    """Train the prediction models."""
    pipeline = UFCPipeline(ctx.obj["config_path"])

    click.echo(f"Loading features from {features}...")
    feature_df = pd.read_parquet(features)
    click.echo(f"  {feature_df.shape[0]} fights, {feature_df.shape[1]} features")

    click.echo("Training models...")
    results = pipeline.train(feature_df)

    pipeline.save(save_model)
    click.echo(f"\nModel saved to {save_model}")

    if "winner_metrics" in results:
        m = results["winner_metrics"]
        click.echo(f"\nTest Set Results:")
        click.echo(f"  Log Loss:  {m['log_loss']:.4f}")
        click.echo(f"  Accuracy:  {m['accuracy']:.1%}")
        click.echo(f"  ECE:       {m['ece']:.4f}")


@cli.command()
@click.option("--fighter-a", required=True, help="Fighter A name")
@click.option("--fighter-b", required=True, help="Fighter B name")
@click.option("--date", default=None, help="Fight date (YYYY-MM-DD)")
@click.option("--model", default="models/pipeline.pkl", help="Path to trained model")
@click.option("--closing-line", type=float, default=None,
              help="Closing line implied probability for fighter A")
@click.option("--rounds", type=int, default=3, help="Scheduled rounds (3 or 5)")
@click.option("--main-event", is_flag=True, help="Is this a main event?")
@click.option("--title-fight", is_flag=True, help="Is this a title fight?")
@click.option("--location", default="", help="Venue location")
@click.option("--json-output", is_flag=True, help="Output as JSON")
@click.pass_context
def predict(ctx, fighter_a, fighter_b, date, model, closing_line,
            rounds, main_event, title_fight, location, json_output):
    """Predict a fight outcome.

    Example:
        ufc-predict predict --fighter-a "Islam Makhachev" --fighter-b "Charles Oliveira" --date 2026-06-01
    """
    click.echo(f"Loading model from {model}...")
    pipeline = UFCPipeline.load(model)

    fight_info = {
        "date": date or "",
        "scheduled_rounds": rounds,
        "is_main_event": main_event or rounds == 5,
        "is_title_fight": title_fight,
        "location": location,
    }

    click.echo(f"\nPredicting: {fighter_a} vs {fighter_b}")
    click.echo("=" * 60)

    result = pipeline.predict_fight(fighter_a, fighter_b, fight_info, closing_line)

    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
        return

    # Pretty print prediction
    click.echo(f"\n{'WINNER PREDICTION':^60}")
    click.echo("-" * 60)
    click.echo(f"  {fighter_a}: {result['win_prob_a']:.1%} "
               f"[90% CI: {result['ci_90_lower']:.1%} - {result['ci_90_upper']:.1%}]")
    click.echo(f"  {fighter_b}: {result['win_prob_b']:.1%}")
    click.echo(f"\n  Predicted Winner: {result['predicted_winner']}")

    if result["edge_vs_line"] is not None:
        edge = result["edge_vs_line"]
        direction = "VALUE" if abs(edge) > 0.05 else "NO EDGE"
        click.echo(f"  Edge vs Line: {edge:+.1%} ({direction})")

    click.echo(f"\n{'METHOD OF VICTORY':^60}")
    click.echo("-" * 60)
    for method, prob in sorted(result["method_distribution"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        click.echo(f"  {method:<12} {prob:>5.1%} {bar}")

    click.echo(f"\n{'ROUND DISTRIBUTION':^60}")
    click.echo("-" * 60)
    for round_name, prob in sorted(result["round_distribution"].items()):
        bar = "█" * int(prob * 30)
        click.echo(f"  {round_name:<10} {prob:>5.1%} {bar}")

    click.echo(f"\n{'FIGHT DURATION':^60}")
    click.echo("-" * 60)
    click.echo(f"  Expected: {result['expected_duration_display']} "
               f"({result['expected_duration_seconds']:.0f} seconds)")

    click.echo(f"\n{'MODEL AGREEMENT':^60}")
    click.echo("-" * 60)
    for model_name, prob in result["model_agreement"].items():
        click.echo(f"  {model_name:<10} {prob:.1%}")

    if result["top_shap_drivers"]:
        click.echo(f"\n{'TOP PREDICTION DRIVERS':^60}")
        click.echo("-" * 60)
        for driver in result["top_shap_drivers"]:
            direction = "↑" if driver["direction"] == "+" else "↓"
            click.echo(f"  {direction} {driver['feature']}: "
                       f"{driver['feature_value']:.3f} (SHAP: {driver['shap_value']:+.3f})")


@cli.command()
@click.option("--features", required=True, help="Path to feature matrix")
@click.option("--model", default="models/pipeline.pkl", help="Path to trained model")
@click.option("--output", default="reports", help="Output directory for report")
@click.pass_context
def evaluate(ctx, features, model, output):
    """Run full evaluation on test set and generate report."""
    pipeline = UFCPipeline.load(model)

    feature_df = pd.read_parquet(features)
    _, _, test_df = pipeline.split_data(feature_df)

    X_test = test_df[pipeline.feature_names].fillna(0)
    pipeline.evaluate(X_test, test_df)

    click.echo(f"\nReport saved to {output}/")


@cli.command()
@click.option("--data", required=True, help="Path to fight data")
@click.option("--save-model", default="models/pipeline.pkl", help="Model output path")
@click.pass_context
def run_full(ctx, data, save_model):
    """Run the full pipeline: features -> train -> evaluate."""
    pipeline = UFCPipeline(ctx.obj["config_path"])

    click.echo("Step 1: Loading data...")
    fights_df = pipeline.load_data(data)
    click.echo(f"  {len(fights_df)} fights")

    click.echo("\nStep 2: Building features...")
    feature_df = pipeline.prepare_features(fights_df)
    click.echo(f"  {feature_df.shape[1]} features per fight")

    click.echo("\nStep 3: Training models...")
    results = pipeline.train(feature_df)

    click.echo("\nStep 4: Saving pipeline...")
    pipeline.save(save_model)

    click.echo("\n✓ Pipeline complete!")


if __name__ == "__main__":
    cli()
