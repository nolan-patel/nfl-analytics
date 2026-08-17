from __future__ import annotations
import io, json, gzip, math, os, sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests

OWNER = "nflverse"
REPO = "nflverse-data"
OUT = Path("data")
OUT.mkdir(exist_ok=True)

# Historical seasons plus the current calendar year.
CURRENT_YEAR = datetime.now(timezone.utc).year
SEASONS = list(range(max(2023, CURRENT_YEAR - 3), CURRENT_YEAR + 1))

session = requests.Session()
session.headers.update({"User-Agent": "nfl-analytics-github-action/1.0"})

def api_json(url: str):
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def release_assets(tag: str):
    rel = api_json(f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{tag}")
    return {a["name"]: a["browser_download_url"] for a in rel["assets"]}

def find_asset(assets: dict[str,str], candidates: list[str], contains: list[str] = []):
    for c in candidates:
        if c in assets:
            return assets[c]
    for name, url in assets.items():
        if all(piece in name for piece in contains):
            return url
    raise FileNotFoundError(f"No matching asset. Tried {candidates}; contains={contains}")

def download_bytes(url: str):
    print("Downloading", url)
    r = session.get(url, timeout=180)
    r.raise_for_status()
    return r.content

def clean_num(v):
    try:
        if pd.isna(v): return 0.0
        return float(v)
    except Exception:
        return 0.0

def safe_div(a,b):
    return float(a/b) if b else 0.0

def team_metadata():
    # nflverse team metadata is small and stable.
    url = "https://raw.githubusercontent.com/nflverse/nflverse-pbp/master/teams_colors_logos.csv"
    df = pd.read_csv(url)
    keep = [c for c in ["team_abbr","team_name","team_color","team_color2","team_logo_espn"] if c in df.columns]
    df = df[keep].drop_duplicates("team_abbr")
    return {r["team_abbr"]: r for r in df.to_dict("records")}

TEAM_META = team_metadata()

def load_pbp(year: int, assets: dict[str,str]):
    url = find_asset(
        assets,
        [f"play_by_play_{year}.csv.gz", f"play_by_play_{year}.csv"],
        [str(year), "play_by_play"]
    )
    raw = download_bytes(url)
    if url.endswith(".gz"):
        raw = gzip.decompress(raw)
    use = [
        "season_type","play_type","posteam","defteam","epa","score_differential",
        "yards_gained","interception","fumble_lost","down","first_down",
        "dropback","sack","qb_hit","qb_scramble"
    ]
    return pd.read_csv(io.BytesIO(raw), usecols=lambda c: c in use, low_memory=False)

def compute_team_metrics(df: pd.DataFrame, year: int):
    df = df[(df.get("season_type") == "REG") & (df.get("play_type").isin(["pass","run"]))].copy()
    numeric = ["epa","score_differential","yards_gained","interception","fumble_lost","down","first_down","dropback","sack","qb_hit","qb_scramble"]
    for c in numeric:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    teams = sorted(set(df["posteam"].dropna()) | set(df["defteam"].dropna()))
    out=[]
    for team in teams:
        o=df[df["posteam"]==team]
        d=df[df["defteam"]==team]
        opass=o[o["play_type"]=="pass"]; orun=o[o["play_type"]=="run"]
        dpass=d[d["play_type"]=="pass"]; drun=d[d["play_type"]=="run"]
        one_o=o[o["score_differential"].abs()<=8]
        one_d=d[d["score_differential"].abs()<=8]
        third_o=o[o["down"]==3]; third_d=d[d["down"]==3]
        drop_d=d[(d["dropback"]==1)|(d["play_type"]=="pass")|(d["sack"]==1)|(d["qb_scramble"]==1)]
        meta=TEAM_META.get(team,{})
        out.append({
            "season":year,"team":team,
            "team_name":meta.get("team_name",team),
            "team_color":meta.get("team_color","#4da3ff"),
            "team_color2":meta.get("team_color2","#91a4bb"),
            "team_logo":meta.get("team_logo_espn",""),
            "off_epa":clean_num(o["epa"].mean()),
            "off_epa_onescore":clean_num(one_o["epa"].mean()),
            # Flip EPA allowed so higher = better defense, matching the dashboard.
            "def_epa":-clean_num(d["epa"].mean()),
            "def_epa_onescore":-clean_num(one_d["epa"].mean()),
            "off_success_rate":safe_div((o["epa"]>0).sum(),len(o)),
            "off_explosive_pass_rate":safe_div((opass["yards_gained"]>=15).sum(),len(opass)),
            "off_explosive_run_rate":safe_div((orun["yards_gained"]>=10).sum(),len(orun)),
            "off_turnover_rate":safe_div(((o["interception"]==1)|(o["fumble_lost"]==1)).sum(),len(o)),
            "off_third_down_conv_rate":safe_div((third_o["first_down"]==1).sum(),len(third_o)),
            "pressure_rate":safe_div(((drop_d["qb_hit"]==1)|(drop_d["sack"]==1)).sum(),len(drop_d)),
            "def_success_rate":safe_div((d["epa"]<0).sum(),len(d)),
            "def_explosive_pass_rate":safe_div((dpass["yards_gained"]>=15).sum(),len(dpass)),
            "def_explosive_run_rate":safe_div((drun["yards_gained"]>=10).sum(),len(drun)),
            "def_turnover_rate":safe_div(((d["interception"]==1)|(d["fumble_lost"]==1)).sum(),len(d)),
            "def_third_down_conv_rate":safe_div((third_d["first_down"]==1).sum(),len(third_d)),
        })
    return out

def map_group(pos):
    p=str(pos or "").upper()
    if p=="QB": return "QB"
    if p in {"RB","FB"}: return "RB"
    if p in {"WR","TE"}: return "WR"
    if p in {"C","G","OG","T","OT","OL","LT","RT","LG","RG"}: return "OL"
    if p in {"CB","DB","S","SS","FS","SAF"}: return "DB"
    if p in {"ILB","MLB","LB"}: return "LB"
    if p in {"DE","DT","NT","DI","DL","OLB","EDGE"}: return "DL"
    return "Other"

def load_players(year:int, assets:dict[str,str]):
    url=find_asset(
        assets,
        [f"stats_player_week_{year}.csv", f"stats_player_week_{year}.csv.gz"],
        [f"stats_player_week_{year}"]
    )
    raw=download_bytes(url)
    if url.endswith(".gz"): raw=gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), low_memory=False)

def weighted(g, val, w):
    if val not in g or w not in g: return 0.0
    ww=pd.to_numeric(g[w],errors="coerce").fillna(0)
    vv=pd.to_numeric(g[val],errors="coerce").fillna(0)
    return safe_div((vv*ww).sum(),ww.sum())

def sumc(g,c):
    return clean_num(pd.to_numeric(g[c],errors="coerce").fillna(0).sum()) if c in g else 0.0

def build_player_metrics(df:pd.DataFrame, year:int):
    if "season_type" in df: df=df[df["season_type"]=="REG"].copy()
    df["position_group_custom"]=df["position"].map(map_group)
    name_col="player_display_name" if "player_display_name" in df else "player_name"
    out=[]
    keys=["team",name_col,"position","position_group_custom"]
    for (team,name,pos,group),g in df.groupby(keys,dropna=False):
        games=max(1,g["week"].nunique() if "week" in g else len(g))
        attempts=sumc(g,"attempts"); completions=sumc(g,"completions"); pass_y=sumc(g,"passing_yards"); pass_td=sumc(g,"passing_tds"); pass_int=sumc(g,"passing_interceptions"); sacks_s=sumc(g,"sacks_suffered"); pass_epa=sumc(g,"passing_epa")
        carries=sumc(g,"carries"); rush_y=sumc(g,"rushing_yards"); rush_td=sumc(g,"rushing_tds"); rush_fd=sumc(g,"rushing_first_downs"); rush_epa=sumc(g,"rushing_epa")
        targets=sumc(g,"targets"); rec_y=sumc(g,"receiving_yards"); rec_td=sumc(g,"receiving_tds"); rec_fd=sumc(g,"receiving_first_downs"); rec_epa=sumc(g,"receiving_epa")
        tackles=sumc(g,"def_tackles_solo")+sumc(g,"def_tackle_assists")
        tfl=sumc(g,"def_tackles_for_loss"); dsacks=sumc(g,"def_sacks"); qbh=sumc(g,"def_qb_hits"); dint=sumc(g,"def_interceptions"); pbu=sumc(g,"def_pass_defended"); ff=sumc(g,"def_fumbles_forced"); dtd=sumc(g,"def_tds")
        pens=sumc(g,"penalties"); peny=sumc(g,"penalty_yards")
        db=attempts+sacks_s
        out.append({
            "season":year,"team":team,"player_name":name,"position":pos,"position_group":group,"games":games,
            "attempts_pg":safe_div(attempts,games),"pass_yards_pg":safe_div(pass_y,games),"pass_tds_pg":safe_div(pass_td,games),"pass_ints_pg":safe_div(pass_int,games),
            "epa_per_dropback":safe_div(pass_epa,db),"yards_per_attempt":safe_div(pass_y,attempts),"completion_pct":safe_div(completions,attempts),"td_rate":safe_div(pass_td,attempts),"int_rate":safe_div(pass_int,attempts),"sack_rate":safe_div(sacks_s,db),"passing_cpoe_avg":weighted(g,"passing_cpoe","attempts"),
            "carries_pg":safe_div(carries,games),"rush_yards_pg":safe_div(rush_y,games),"rush_tds_pg":safe_div(rush_td,games),"yards_per_carry":safe_div(rush_y,carries),"epa_per_rush":safe_div(rush_epa,carries),"rush_fd_rate":safe_div(rush_fd,carries),
            "targets_pg":safe_div(targets,games),"rec_yards_pg":safe_div(rec_y,games),"rec_tds_pg":safe_div(rec_td,games),"yards_per_target":safe_div(rec_y,targets),"epa_per_target":safe_div(rec_epa,targets),"rec_fd_rate":safe_div(rec_fd,targets),
            "target_share_avg":weighted(g,"target_share","targets"),"air_yards_share_avg":weighted(g,"air_yards_share","targets"),"wopr_avg":weighted(g,"wopr","targets"),"racr_avg":weighted(g,"racr","targets"),
            "tackles_pg":safe_div(tackles,games),"tfl_pg":safe_div(tfl,games),"sacks_pg":safe_div(dsacks,games),"qb_hits_pg":safe_div(qbh,games),"ints_pg":safe_div(dint,games),"pbu_pg":safe_div(pbu,games),"forced_fumbles_pg":safe_div(ff,games),"def_tds_pg":safe_div(dtd,games),
            "penalties_pg":safe_div(pens,games),"penalty_yards_pg":safe_div(peny,games)
        })
    return out

def write_json(path:Path,obj):
    path.write_text(json.dumps(obj,separators=(",",":"),allow_nan=False),encoding="utf-8")

def main():
    pbp_assets=release_assets("pbp")
    player_assets=release_assets("stats_player")
    good=[]
    for year in SEASONS:
        try:
            print(f"\n=== {year} ===")
            pbp=load_pbp(year,pbp_assets)
            teams=compute_team_metrics(pbp,year)
            players=build_player_metrics(load_players(year,player_assets),year)
            if not teams or not players:
                raise RuntimeError("Source data returned no records")
            write_json(OUT/f"team_metrics_{year}.json",teams)
            write_json(OUT/f"player_metrics_{year}.json",players)
            good.append(year)
            print(f"Wrote {len(teams)} team rows and {len(players)} player rows")
        except Exception as e:
            print(f"Skipping {year}: {e}", file=sys.stderr)
    if not good:
        raise RuntimeError("No seasons could be generated")
    now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    write_json(OUT/"manifest.json",{"seasons":sorted(good,reverse=True),"default_season":max(good),"updated_utc":now})
    print("Available seasons:",good)

if __name__=="__main__":
    main()
