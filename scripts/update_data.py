from __future__ import annotations

import gzip
import io
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


OUT = Path("data")
OUT.mkdir(exist_ok=True)

CURRENT_YEAR = datetime.now(timezone.utc).year

# We want the completed seasons plus the current season when available.
SEASONS = list(range(2023, CURRENT_YEAR + 1))

session = requests.Session()
session.headers.update(
    {
        "User-Agent": "nfl-analytics-dashboard/1.0",
        "Accept": "application/vnd.github+json",
    }
)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def safe_div(num, den):
    den = safe_float(den)

    if den == 0:
        return 0.0

    return safe_float(num) / den


def json_clean(value):
    """
    Recursively convert pandas / numpy / NaN values
    into JSON-safe Python objects.
    """

    if isinstance(value, dict):
        return {
            str(k): json_clean(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [json_clean(v) for v in value]

    if isinstance(value, tuple):
        return [json_clean(v) for v in value]

    # Handle pandas / numpy scalar values
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def write_json(path, obj):
    cleaned = json_clean(obj)

    path.write_text(
        json.dumps(
            cleaned,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def download(url):
    print(f"Downloading {url}")

    response = session.get(
        url,
        timeout=240,
        allow_redirects=True,
    )

    response.raise_for_status()

    return response.content


def github_release_assets(tag):
    url = (
        "https://api.github.com/repos/"
        "nflverse/nflverse-data/releases/tags/"
        f"{tag}"
    )

    response = session.get(url, timeout=60)
    response.raise_for_status()

    release = response.json()

    return {
        asset["name"]: asset["browser_download_url"]
        for asset in release.get("assets", [])
    }


# ---------------------------------------------------------
# NFL TEAM METADATA
# ---------------------------------------------------------

def load_team_metadata():

    url = (
        "https://raw.githubusercontent.com/"
        "nflverse/nflverse-pbp/master/"
        "teams_colors_logos.csv"
    )

    print("Loading team metadata...")

    df = pd.read_csv(url)

    result = {}

    for _, row in df.iterrows():

        team = str(row.get("team_abbr", "")).strip()

        if not team:
            continue

        result[team] = {
            "team_name": (
                str(row.get("team_name"))
                if pd.notna(row.get("team_name"))
                else team
            ),
            "team_color": (
                str(row.get("team_color"))
                if pd.notna(row.get("team_color"))
                else "#4da3ff"
            ),
            "team_color2": (
                str(row.get("team_color2"))
                if pd.notna(row.get("team_color2"))
                else "#91a4bb"
            ),
            "team_logo": (
                str(row.get("team_logo_espn"))
                if pd.notna(row.get("team_logo_espn"))
                else ""
            ),
        }

    return result


TEAM_META = load_team_metadata()


# ---------------------------------------------------------
# PLAY BY PLAY
# ---------------------------------------------------------

def find_pbp_asset(assets, season):

    preferred = [
        f"play_by_play_{season}.csv.gz",
        f"play_by_play_{season}.csv",
    ]

    for name in preferred:
        if name in assets:
            return assets[name]

    for name, url in assets.items():

        name_lower = name.lower()

        if (
            str(season) in name_lower
            and "play_by_play" in name_lower
            and (
                name_lower.endswith(".csv")
                or name_lower.endswith(".csv.gz")
            )
        ):
            return url

    raise FileNotFoundError(
        f"No play-by-play CSV found for {season}"
    )


def load_pbp(assets, season):

    url = find_pbp_asset(assets, season)

    raw = download(url)

    if url.endswith(".gz"):
        raw = gzip.decompress(raw)

    wanted = {
        "season_type",
        "play_type",
        "posteam",
        "defteam",
        "epa",
        "score_differential",
        "yards_gained",
        "interception",
        "fumble_lost",
        "down",
        "first_down",
        "dropback",
        "sack",
        "qb_hit",
        "qb_scramble",
    }

    df = pd.read_csv(
        io.BytesIO(raw),
        usecols=lambda col: col in wanted,
        low_memory=False,
    )

    return df


def compute_team_metrics(df, season):

    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"].copy()

    df = df[
        df["play_type"].isin(["pass", "run"])
    ].copy()

    numeric_cols = [
        "epa",
        "score_differential",
        "yards_gained",
        "interception",
        "fumble_lost",
        "down",
        "first_down",
        "dropback",
        "sack",
        "qb_hit",
        "qb_scramble",
    ]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            ).fillna(0)

    teams = sorted(
        set(df["posteam"].dropna().astype(str))
        |
        set(df["defteam"].dropna().astype(str))
    )

    output = []

    for team in teams:

        offense = df[df["posteam"] == team]
        defense = df[df["defteam"] == team]

        if offense.empty and defense.empty:
            continue

        off_pass = offense[
            offense["play_type"] == "pass"
        ]

        off_run = offense[
            offense["play_type"] == "run"
        ]

        def_pass = defense[
            defense["play_type"] == "pass"
        ]

        def_run = defense[
            defense["play_type"] == "run"
        ]

        one_off = offense[
            offense["score_differential"].abs() <= 8
        ]

        one_def = defense[
            defense["score_differential"].abs() <= 8
        ]

        third_off = offense[
            offense["down"] == 3
        ]

        third_def = defense[
            defense["down"] == 3
        ]

        dropbacks = defense[
            (defense["dropback"] == 1)
            |
            (defense["play_type"] == "pass")
            |
            (defense["sack"] == 1)
            |
            (defense["qb_scramble"] == 1)
        ]

        metadata = TEAM_META.get(team, {})

        row = {

            "season": int(season),

            "team": team,

            "team_name":
                metadata.get("team_name", team),

            "team_color":
                metadata.get(
                    "team_color",
                    "#4da3ff",
                ),

            "team_color2":
                metadata.get(
                    "team_color2",
                    "#91a4bb",
                ),

            "team_logo":
                metadata.get(
                    "team_logo",
                    "",
                ),

            # OFFENSE

            "off_epa":
                safe_float(
                    offense["epa"].mean()
                ),

            "off_epa_onescore":
                safe_float(
                    one_off["epa"].mean()
                ),

            "off_success_rate":
                safe_div(
                    (offense["epa"] > 0).sum(),
                    len(offense),
                ),

            "off_explosive_pass_rate":
                safe_div(
                    (
                        off_pass["yards_gained"]
                        >= 15
                    ).sum(),
                    len(off_pass),
                ),

            "off_explosive_run_rate":
                safe_div(
                    (
                        off_run["yards_gained"]
                        >= 10
                    ).sum(),
                    len(off_run),
                ),

            "off_turnover_rate":
                safe_div(
                    (
                        (offense["interception"] == 1)
                        |
                        (offense["fumble_lost"] == 1)
                    ).sum(),
                    len(offense),
                ),

            "off_third_down_conv_rate":
                safe_div(
                    (
                        third_off["first_down"]
                        == 1
                    ).sum(),
                    len(third_off),
                ),

            # DEFENSE

            # Flip sign so higher = better defense
            "def_epa":
                -safe_float(
                    defense["epa"].mean()
                ),

            "def_epa_onescore":
                -safe_float(
                    one_def["epa"].mean()
                ),

            "def_success_rate":
                safe_div(
                    (
                        defense["epa"]
                        < 0
                    ).sum(),
                    len(defense),
                ),

            "pressure_rate":
                safe_div(
                    (
                        (dropbacks["qb_hit"] == 1)
                        |
                        (dropbacks["sack"] == 1)
                    ).sum(),
                    len(dropbacks),
                ),

            "def_explosive_pass_rate":
                safe_div(
                    (
                        def_pass["yards_gained"]
                        >= 15
                    ).sum(),
                    len(def_pass),
                ),

            "def_explosive_run_rate":
                safe_div(
                    (
                        def_run["yards_gained"]
                        >= 10
                    ).sum(),
                    len(def_run),
                ),

            "def_turnover_rate":
                safe_div(
                    (
                        (defense["interception"] == 1)
                        |
                        (defense["fumble_lost"] == 1)
                    ).sum(),
                    len(defense),
                ),

            "def_third_down_conv_rate":
                safe_div(
                    (
                        third_def["first_down"]
                        == 1
                    ).sum(),
                    len(third_def),
                ),
        }

        output.append(row)

    return output


# ---------------------------------------------------------
# PLAYER DATA
# ---------------------------------------------------------

def find_player_asset(assets, season):

    preferred = [
        f"stats_player_week_{season}.csv",
        f"stats_player_week_{season}.csv.gz",
    ]

    for name in preferred:

        if name in assets:
            return assets[name]

    for name, url in assets.items():

        low = name.lower()

        if (
            str(season) in low
            and "stats_player_week" in low
            and (
                low.endswith(".csv")
                or low.endswith(".csv.gz")
            )
        ):
            return url

    raise FileNotFoundError(
        f"No weekly player CSV found for {season}"
    )


def load_players(assets, season):

    url = find_player_asset(
        assets,
        season,
    )

    raw = download(url)

    if url.endswith(".gz"):
        raw = gzip.decompress(raw)

    df = pd.read_csv(
        io.BytesIO(raw),
        low_memory=False,
    )

    return df


def map_position(position):

    pos = str(position).upper()

    if pos == "QB":
        return "QB"

    if pos in {"RB", "FB"}:
        return "RB"

    if pos in {"WR", "TE"}:
        return "WR"

    if pos in {
        "C",
        "G",
        "OG",
        "T",
        "OT",
        "OL",
        "LT",
        "RT",
        "LG",
        "RG",
    }:
        return "OL"

    if pos in {
        "CB",
        "DB",
        "S",
        "SS",
        "FS",
        "SAF",
    }:
        return "DB"

    if pos in {
        "ILB",
        "MLB",
        "LB",
    }:
        return "LB"

    if pos in {
        "DE",
        "DT",
        "NT",
        "DI",
        "DL",
        "OLB",
        "EDGE",
    }:
        return "DL"

    return "Other"


def column_sum(group, column):

    if column not in group.columns:
        return 0.0

    return safe_float(
        pd.to_numeric(
            group[column],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )


def weighted_average(
    group,
    value_column,
    weight_column,
):

    if (
        value_column not in group.columns
        or weight_column not in group.columns
    ):
        return 0.0

    values = pd.to_numeric(
        group[value_column],
        errors="coerce",
    ).fillna(0)

    weights = pd.to_numeric(
        group[weight_column],
        errors="coerce",
    ).fillna(0)

    return safe_div(
        (values * weights).sum(),
        weights.sum(),
    )


def build_player_metrics(df, season):

    if "season_type" in df.columns:
        df = df[
            df["season_type"] == "REG"
        ].copy()

    df["position_group_custom"] = (
        df["position"]
        .apply(map_position)
    )

    if "player_display_name" in df.columns:
        name_col = "player_display_name"

    elif "player_name" in df.columns:
        name_col = "player_name"

    else:
        name_col = "player_id"

    # Remove rows that cannot represent a real player.
    df = df[
        df[name_col].notna()
        & df["team"].notna()
        & df["position"].notna()
    ].copy()

    results = []

    group_columns = [
        "team",
        name_col,
        "position",
        "position_group_custom",
    ]

    for (
        team,
        player_name,
        position,
        position_group,
    ), group in df.groupby(group_columns):

        if "week" in group.columns:
            games = safe_int(
                group["week"].nunique(),
                1,
            )
        else:
            games = len(group)

        games = max(1, games)

        # PASSING

        completions = column_sum(
            group,
            "completions",
        )

        attempts = column_sum(
            group,
            "attempts",
        )

        pass_yards = column_sum(
            group,
            "passing_yards",
        )

        pass_tds = column_sum(
            group,
            "passing_tds",
        )

        pass_ints = column_sum(
            group,
            "passing_interceptions",
        )

        sacks_suffered = column_sum(
            group,
            "sacks_suffered",
        )

        passing_epa = column_sum(
            group,
            "passing_epa",
        )

        dropbacks = (
            attempts
            + sacks_suffered
        )

        # RUSHING

        carries = column_sum(
            group,
            "carries",
        )

        rush_yards = column_sum(
            group,
            "rushing_yards",
        )

        rush_tds = column_sum(
            group,
            "rushing_tds",
        )

        rush_first_downs = column_sum(
            group,
            "rushing_first_downs",
        )

        rush_epa = column_sum(
            group,
            "rushing_epa",
        )

        # RECEIVING

        targets = column_sum(
            group,
            "targets",
        )

        receiving_yards = column_sum(
            group,
            "receiving_yards",
        )

        receiving_tds = column_sum(
            group,
            "receiving_tds",
        )

        receiving_first_downs = (
            column_sum(
                group,
                "receiving_first_downs",
            )
        )

        receiving_epa = column_sum(
            group,
            "receiving_epa",
        )

        # DEFENSE

        tackles = (
            column_sum(
                group,
                "def_tackles_solo",
            )
            +
            column_sum(
                group,
                "def_tackle_assists",
            )
        )

        tfl = column_sum(
            group,
            "def_tackles_for_loss",
        )

        defensive_sacks = column_sum(
            group,
            "def_sacks",
        )

        qb_hits = column_sum(
            group,
            "def_qb_hits",
        )

        defensive_ints = column_sum(
            group,
            "def_interceptions",
        )

        pass_defended = column_sum(
            group,
            "def_pass_defended",
        )

        forced_fumbles = column_sum(
            group,
            "def_fumbles_forced",
        )

        defensive_tds = column_sum(
            group,
            "def_tds",
        )

        penalties = column_sum(
            group,
            "penalties",
        )

        penalty_yards = column_sum(
            group,
            "penalty_yards",
        )

        row = {

            "season": int(season),

            "team": str(team),

            "player_name":
                str(player_name),

            "position":
                str(position),

            "position_group":
                str(position_group),

            "games":
                int(games),

            # QB

            "attempts_pg":
                safe_div(
                    attempts,
                    games,
                ),

            "pass_yards_pg":
                safe_div(
                    pass_yards,
                    games,
                ),

            "pass_tds_pg":
                safe_div(
                    pass_tds,
                    games,
                ),

            "pass_ints_pg":
                safe_div(
                    pass_ints,
                    games,
                ),

            "epa_per_dropback":
                safe_div(
                    passing_epa,
                    dropbacks,
                ),

            "yards_per_attempt":
                safe_div(
                    pass_yards,
                    attempts,
                ),

            "completion_pct":
                safe_div(
                    completions,
                    attempts,
                ),

            "td_rate":
                safe_div(
                    pass_tds,
                    attempts,
                ),

            "int_rate":
                safe_div(
                    pass_ints,
                    attempts,
                ),

            "sack_rate":
                safe_div(
                    sacks_suffered,
                    dropbacks,
                ),

            "passing_cpoe_avg":
                weighted_average(
                    group,
                    "passing_cpoe",
                    "attempts",
                ),

            # RB

            "carries_pg":
                safe_div(
                    carries,
                    games,
                ),

            "rush_yards_pg":
                safe_div(
                    rush_yards,
                    games,
                ),

            "rush_tds_pg":
                safe_div(
                    rush_tds,
                    games,
                ),

            "yards_per_carry":
                safe_div(
                    rush_yards,
                    carries,
                ),

            "epa_per_rush":
                safe_div(
                    rush_epa,
                    carries,
                ),

            "rush_fd_rate":
                safe_div(
                    rush_first_downs,
                    carries,
                ),

            # RECEIVING

            "targets_pg":
                safe_div(
                    targets,
                    games,
                ),

            "rec_yards_pg":
                safe_div(
                    receiving_yards,
                    games,
                ),

            "rec_tds_pg":
                safe_div(
                    receiving_tds,
                    games,
                ),

            "yards_per_target":
                safe_div(
                    receiving_yards,
                    targets,
                ),

            "epa_per_target":
                safe_div(
                    receiving_epa,
                    targets,
                ),

            "rec_fd_rate":
                safe_div(
                    receiving_first_downs,
                    targets,
                ),

            "target_share_avg":
                weighted_average(
                    group,
                    "target_share",
                    "targets",
                ),

            "air_yards_share_avg":
                weighted_average(
                    group,
                    "air_yards_share",
                    "targets",
                ),

            "wopr_avg":
                weighted_average(
                    group,
                    "wopr",
                    "targets",
                ),

            "racr_avg":
                weighted_average(
                    group,
                    "racr",
                    "targets",
                ),

            # DEFENSE

            "tackles_pg":
                safe_div(
                    tackles,
                    games,
                ),

            "tfl_pg":
                safe_div(
                    tfl,
                    games,
                ),

            "sacks_pg":
                safe_div(
                    defensive_sacks,
                    games,
                ),

            "qb_hits_pg":
                safe_div(
                    qb_hits,
                    games,
                ),

            "ints_pg":
                safe_div(
                    defensive_ints,
                    games,
                ),

            "pbu_pg":
                safe_div(
                    pass_defended,
                    games,
                ),

            "forced_fumbles_pg":
                safe_div(
                    forced_fumbles,
                    games,
                ),

            "def_tds_pg":
                safe_div(
                    defensive_tds,
                    games,
                ),

            # OL

            "penalties_pg":
                safe_div(
                    penalties,
                    games,
                ),

            "penalty_yards_pg":
                safe_div(
                    penalty_yards,
                    games,
                ),
        }

        results.append(row)

    return results


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    print("Loading nflverse release information...")

    pbp_assets = github_release_assets(
        "pbp"
    )

    player_assets = github_release_assets(
        "stats_player"
    )

    successful_seasons = []

    for season in SEASONS:

        print()
        print("=" * 50)
        print(f"SEASON {season}")
        print("=" * 50)

        try:

            print("Loading play-by-play...")

            pbp = load_pbp(
                pbp_assets,
                season,
            )

            print(
                f"Play-by-play rows: {len(pbp):,}"
            )

            team_metrics = (
                compute_team_metrics(
                    pbp,
                    season,
                )
            )

            print(
                f"Team metric rows: "
                f"{len(team_metrics)}"
            )

            print("Loading player stats...")

            player_df = load_players(
                player_assets,
                season,
            )

            print(
                f"Weekly player rows: "
                f"{len(player_df):,}"
            )

            player_metrics = (
                build_player_metrics(
                    player_df,
                    season,
                )
            )

            print(
                f"Player metric rows: "
                f"{len(player_metrics):,}"
            )

            if not team_metrics:
                raise RuntimeError(
                    "No team metrics generated"
                )

            if not player_metrics:
                raise RuntimeError(
                    "No player metrics generated"
                )

            write_json(
                OUT
                / f"team_metrics_{season}.json",
                team_metrics,
            )

            write_json(
                OUT
                / f"player_metrics_{season}.json",
                player_metrics,
            )

            successful_seasons.append(
                int(season)
            )

            print(
                f"✅ Completed {season}"
            )

        except Exception as exc:

            print(
                f"❌ Skipping {season}: "
                f"{type(exc).__name__}: {exc}"
            )

            continue

    if not successful_seasons:

        raise RuntimeError(
            "No seasons could be generated"
        )

    successful_seasons = sorted(
        successful_seasons,
        reverse=True,
    )

    manifest = {

        "seasons":
            successful_seasons,

        "default_season":
            successful_seasons[0],

        "updated_utc":
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
    }

    write_json(
        OUT / "manifest.json",
        manifest,
    )

    print()
    print("================================")
    print("DONE")
    print("================================")

    print(
        "Available seasons:",
        successful_seasons,
    )


if __name__ == "__main__":
    main()
