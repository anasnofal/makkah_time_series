from pathlib import Path
import os
import pandas as pd


def find_project_root(
    env_var="MAKKAH_PROJECT_ROOT", target="data/NLTP_F_DAILY_Makkah.csv", max_levels=8
):
    """Return the project root path.

    Preference order:
      1) environment variable MAKKAH_PROJECT_ROOT
      2) walk up from script location looking for data/target
      3) fallback to cwd
    """
    # 1) env var
    env = os.environ.get(env_var)
    if env:
        return str(Path(env).resolve())

    # 2) walk up from this file's parent
    try:
        p = Path(__file__).resolve().parent
    except NameError:
        p = Path.cwd().resolve()

    for _ in range(max_levels):
        candidate = p / "data" / Path(target).name
        if candidate.exists():
            return str(p)
        if p.parent == p:
            break
        p = p.parent

    # 3) fallback
    return str(Path.cwd())


# Resolve directories once
ROOT = find_project_root()
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)


def load_power(power_csv=None):
    path = power_csv or os.path.join(DATA_DIR, "NLTP_F_DAILY_Makkah.csv")
    power = pd.read_csv(path)
    # keep only date and the power value
    if "System_City_Calculated" in power.columns:
        power = power[["Date", "System_City_Calculated"]].rename(
            columns={"System_City_Calculated": "power"}
        )
    elif "power" in power.columns:
        power = power[["Date", "power"]]
    power["Date"] = pd.to_datetime(power["Date"])
    return power


def load_pop(pop_csv=None):
    path = pop_csv or os.path.join(DATA_DIR, "pop - csvData.csv")
    pop = pd.read_csv(path)
    pop = pop.drop(columns=["population", "growthRate"], errors="ignore")
    return pop


def load_temp(temp_csv=None):
    path = temp_csv or os.path.join(DATA_DIR, "temp.csv")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    # keep other columns as-is
    return df


def load_and_merge(power_csv=None, temp_csv=None, pop_csv=None, save_merged=False):
    """Load power, temp and pop tables and merge them by Date/year.

    The merge strategy:
      - left join temp (df) with power on 'Date' so each temp row gets matching power or NaN
      - merge population by year
    """
    power = load_power(power_csv)
    pop = load_pop(pop_csv)
    df = load_temp(temp_csv)

    # Merge power by Date (aligns by date rather than by position)
    merged = df.merge(power[["Date", "power"]], on="Date", how="left")

    # Merge population by year
    merged["year"] = merged["Date"].dt.year
    merged = merged.merge(pop, on="year", how="left")
    merged = merged.drop(columns=["year"], errors="ignore")
    # rename population column if present
    if "Makkah" in merged.columns:
        merged = merged.rename(columns={"Makkah": "Population"})

    # Adjust temp like in original notebook
    if "temp" in merged.columns:
        merged["temp"] = merged["temp"] - 22

    if save_merged:
        out_path = os.path.join(OUT_DIR, "merged_data.csv")
        merged.to_csv(out_path, index=False)
    return merged
