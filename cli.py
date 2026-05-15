"""Unified CLI for multi-sport prediction.

Usage:
    python cli.py --sport ufc scrape --source ufcstats
    python cli.py --sport ufc run-full --data data/raw/ufcstats_fights.csv
    python cli.py --sport ufc predict --fighter-a "Islam Makhachev" --fighter-b "Charles Oliveira"

    python cli.py --sport soccer scrape
    python cli.py --sport soccer run-full --data data/raw/world_cup_matches.csv
    python cli.py --sport soccer predict --team-a "Brazil" --team-b "France"

    python cli.py --sport mlb scrape
    python cli.py --sport mlb run-full --data data/raw/mlb_games.csv
    python cli.py --sport mlb predict --team-a "NYY" --team-b "BOS"
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


SPORT_REGISTRY = {
    "ufc": {
        "pipeline_cls": "ufc_predict.pipeline.UFCPipeline",
        "default_data": "data/raw/ufcstats_fights.csv",
        "default_model": "models/ufc_pipeline.pkl",
        "entity_names": ("fighter-a", "fighter-b"),
    },
    "soccer": {
        "pipeline_cls": "soccer_predict.pipeline.SoccerPipeline",
        "default_data": "data/raw/world_cup_matches.csv",
        "default_model": "models/soccer_pipeline.pkl",
        "entity_names": ("team-a", "team-b"),
    },
    "mlb": {
        "pipeline_cls": "mlb_predict.pipeline.MLBPipeline",
        "default_data": "data/raw/mlb_games.csv",
        "default_model": "models/mlb_pipeline.pkl",
        "entity_names": ("team-a", "team-b"),
    },
}


def _import_pipeline(sport: str):
    """Dynamically import the pipeline class for a sport."""
    info = SPORT_REGISTRY[sport]
    module_path, cls_name = info["pipeline_cls"].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


@click.group()
@click.option("--sport", type=click.Choice(["ufc", "soccer", "mlb"]),
              default="ufc", help="Sport to predict")
@click.option("--config", default=None, help="Path to config YAML")
@click.option("--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, sport, config, verbose):
    """Multi-Sport Prediction System.

    Supports UFC, World Cup Soccer, and MLB predictions.
    """
    ctx.ensure_object(dict)
    ctx.obj["sport"] = sport
    ctx.obj["config_path"] = config
    ctx.obj["registry"] = SPORT_REGISTRY[sport]
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option("--output", default="data/raw", help="Output directory")
@click.pass_context
def scrape(ctx, output):
    """Scrape data for the selected sport."""
    sport = ctx.obj["sport"]
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import time
    start = time.time()

    if sport == "ufc":
        from ufc_predict.scrapers import UFCStatsScraper
        click.echo("Scraping UFC Stats...")
        scraper = UFCStatsScraper(cache_dir="data/cache")
        df = scraper.scrape_all()
        out_path = output_dir / "ufcstats_fights.csv"
        df.to_csv(out_path, index=False)
        click.echo(f"  Saved {len(df)} fights to {out_path}")

    elif sport == "soccer":
        from soccer_predict.scrapers import FBrefScraper, FIFARankingsScraper, ClubLeagueScraper
        click.echo("Scraping FBref World Cup matches...")
        scraper = FBrefScraper(cache_dir="data/cache")
        df = scraper.scrape_all()
        out_path = output_dir / "world_cup_matches.csv"
        df.to_csv(out_path, index=False)
        click.echo(f"  Saved {len(df)} matches to {out_path}")

        click.echo("Scraping FIFA Rankings...")
        rankings = FIFARankingsScraper(cache_dir="data/cache")
        rank_df = rankings.scrape_all()
        rank_path = output_dir / "fifa_rankings.csv"
        rank_df.to_csv(rank_path, index=False)
        click.echo(f"  Saved {len(rank_df)} ranking entries to {rank_path}")

        click.echo("Scraping top-5 European club leagues (PL, La Liga, BuLi, Serie A, Ligue 1)...")
        club_scraper = ClubLeagueScraper(cache_dir="data/cache")
        club_df = club_scraper.scrape_all_leagues()
        club_path = output_dir / "club_league_matches.csv"
        club_df.to_csv(club_path, index=False)
        click.echo(f"  Saved {len(club_df)} club matches to {club_path}")

    elif sport == "mlb":
        from mlb_predict.scrapers import BaseballReferenceScraper, FanGraphsScraper
        click.echo("Scraping Baseball Reference...")
        scraper = BaseballReferenceScraper(cache_dir="data/cache")
        df = scraper.scrape_all()
        out_path = output_dir / "mlb_games.csv"
        df.to_csv(out_path, index=False)
        click.echo(f"  Saved {len(df)} games to {out_path}")

        click.echo("Scraping FanGraphs pitcher stats...")
        fg = FanGraphsScraper(cache_dir="data/cache")
        fg_df = fg.scrape_all()
        fg_path = output_dir / "fangraphs_pitchers.csv"
        fg_df.to_csv(fg_path, index=False)
        click.echo(f"  Saved {len(fg_df)} pitcher records to {fg_path}")

    elapsed = time.time() - start
    click.echo(f"\nDone! ({elapsed:.0f}s)")


@cli.command()
@click.option("--data", default=None, help="Path to event data")
@click.option("--save-model", default=None, help="Model output path")
@click.pass_context
def run_full(ctx, data, save_model):
    """Run the full pipeline: load -> features -> train -> save."""
    sport = ctx.obj["sport"]
    reg = ctx.obj["registry"]
    data = data or reg["default_data"]
    save_model = save_model or reg["default_model"]

    PipelineCls = _import_pipeline(sport)
    pipeline = PipelineCls(ctx.obj["config_path"])

    click.echo(f"[{sport.upper()}] Step 1: Loading data from {data}...")
    events_df = pipeline.load_data(data)
    click.echo(f"  {len(events_df)} events")

    click.echo(f"\n[{sport.upper()}] Step 2: Building features...")
    feature_df = pipeline.prepare_features(events_df)
    click.echo(f"  {feature_df.shape[1]} features per event")

    click.echo(f"\n[{sport.upper()}] Step 3: Training models...")
    results = pipeline.train(feature_df)

    for key, val in results.items():
        if isinstance(val, float):
            click.echo(f"  {key}: {val:.4f}")
        else:
            click.echo(f"  {key}: {val}")

    click.echo(f"\n[{sport.upper()}] Step 4: Saving to {save_model}...")
    pipeline.save(save_model)
    click.echo("Pipeline complete!")


@cli.command()
@click.option("--entity-a", required=True, help="Entity A (fighter/team)")
@click.option("--entity-b", required=True, help="Entity B (fighter/team)")
@click.option("--date", default=None, help="Event date (YYYY-MM-DD)")
@click.option("--model", default=None, help="Path to trained model")
@click.pass_context
def predict(ctx, entity_a, entity_b, date, model):
    """Predict a matchup outcome."""
    sport = ctx.obj["sport"]
    reg = ctx.obj["registry"]
    model_path = model or reg["default_model"]

    PipelineCls = _import_pipeline(sport)

    click.echo(f"Loading {sport.upper()} model from {model_path}...")
    pipeline = PipelineCls.load(model_path)

    context = {}
    if date:
        context["date"] = date

    click.echo(f"\n{'=' * 60}")
    click.echo(f"  {entity_a}  vs  {entity_b}")
    click.echo(f"{'=' * 60}")

    if sport == "ufc":
        # UFC uses the existing predict_fight interface
        fight_info = {"date": date or ""}
        result = pipeline.predict_fight(entity_a, entity_b, fight_info)

        click.echo(f"\n  {entity_a}: {result['win_prob_a']:.1%}")
        click.echo(f"  {entity_b}: {result['win_prob_b']:.1%}")
        click.echo(f"  Predicted Winner: {result['predicted_winner']}")

    elif sport == "soccer":
        result = pipeline.predict_matchup(entity_a, entity_b, context)

        click.echo(f"\n  {entity_a} Win:  {result['home_win_prob']:.1%}")
        click.echo(f"  Draw:         {result['draw_prob']:.1%}")
        click.echo(f"  {entity_b} Win:  {result['away_win_prob']:.1%}")
        click.echo(f"  Predicted: {result['predicted_outcome']}")

    elif sport == "mlb":
        result = pipeline.predict_matchup(entity_a, entity_b, context)

        click.echo(f"\n  {entity_a} (Home): {result['home_win_prob']:.1%}")
        click.echo(f"  {entity_b} (Away): {result['away_win_prob']:.1%}")
        click.echo(f"  Predicted Winner: {result['predicted_winner']}")

    click.echo("")


@cli.command()
@click.option("--source-dir", default=None, help="Directory of articles to ingest")
@click.option("--rss-url", default=None, help="RSS feed URL to ingest")
@click.pass_context
def ingest(ctx, source_dir, rss_url):
    """Ingest articles into RAG vector store."""
    sport = ctx.obj["sport"]

    if sport == "ufc":
        # Use existing UFC ingest
        from ufc_predict.rag.store import FightDocumentStore
        from ufc_predict.rag.ingest import ArticleIngester
    else:
        click.echo(f"RAG ingestion for {sport} uses the same system as UFC.")
        click.echo("Ensure articles reference the relevant entities (teams/players).")
        from ufc_predict.rag.store import FightDocumentStore
        from ufc_predict.rag.ingest import ArticleIngester

    store = FightDocumentStore(persist_dir=f"data/vectorstore_{sport}")
    ingester = ArticleIngester(store=store)

    total = 0
    if source_dir:
        click.echo(f"Ingesting from directory: {source_dir}")
        total += ingester.ingest_from_directory(source_dir)

    if rss_url:
        click.echo(f"Ingesting from RSS: {rss_url}")
        total += ingester.ingest_from_rss(rss_url)

    if not source_dir and not rss_url:
        click.echo("Provide --source-dir or --rss-url")
        return

    click.echo(f"\nIngested {total} chunks. Store has {store.count()} total.")


if __name__ == "__main__":
    cli()
