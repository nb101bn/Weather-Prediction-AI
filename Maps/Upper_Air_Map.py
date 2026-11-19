"""
Upper_Air_Map.py
================
Isoplethed constant-pressure chart (default 300 mb) from the model's 3 km fields:

  * wind speed  - shaded isotachs (kt)
  * wind        - barbs (kt), thinned to a readable spacing
  * heights     - solid black contours of geopotential height (dam). On a constant-PRESSURE chart the
                  "pressure lines" are height contours (isohypses): pressure is the same everywhere on
                  the 300 mb surface, and heights play the role isobars play on a surface map.
  * divergence  - magenta solid contours (divergence) / dashed blue (convergence), 10^-5 s^-1,
                  computed from the model winds and lightly smoothed (3 km divergence is very noisy)

Uses cartopy (Lambert conformal, same projection as the grid, so grid-relative barbs are drawn in
native coordinates) with state/coast outlines when Natural Earth data can be loaded; falls back to
lat/lon gridlines only (offline), or to plain matplotlib if cartopy is not installed.
"""
from __future__ import annotations

import datetime
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from Models import Atmos_Physics as ap
from Models.Model_Grid import LambertGrid

KT = 1.943844
ISOTACH_LEVELS = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200]
ISOTACH_COLORS = ["#dbe9f6", "#bad6eb", "#89bedc", "#539ecd", "#2b7bba", "#0b559f", "#2f2a8c", "#54278f",
                  "#7a52a5", "#9e9ac8", "#c994c7", "#df65b0", "#e7298a", "#ce1256"]
DIVERGENCE_LEVELS = [2, 4, 6, 8, 10, 15, 20, 30, 40, 60]


def _basemap_available() -> bool:
    try:
        import cartopy.io.shapereader as shpreader
        shpreader.natural_earth(resolution="50m", category="cultural", name="admin_1_states_provinces_lakes")
        shpreader.natural_earth(resolution="50m", category="physical", name="coastline")
        return True
    except Exception:
        return False


_BASEMAP = None


def compute_map_fields(grid: LambertGrid, z, u, v, div_smooth_km: float = 20.0):
    """Heights (dam), speed (kt), divergence (1e-5 s^-1, smoothed) from 2-D level fields (grid-relative winds)."""
    m = torch.as_tensor(grid.map_factor(), dtype=torch.float64)
    ut, vt = torch.as_tensor(u, dtype=torch.float64), torch.as_tensor(v, dtype=torch.float64)
    div = ap.divergence(ut, vt, m, grid.dx)
    div = ap.gaussian_smooth(div[None], div_smooth_km * 1000.0 / grid.dx / 2.0)[0] if div_smooth_km else div
    return {"height_dam": np.asarray(z) / 10.0, "speed_kt": np.hypot(u, v) * KT,
            "divergence_1e5": div.numpy() * 1e5}


def render_upper_air_map(grid: LambertGrid, z, u, v, out_path: str, level_hpa: int = 300,
                         init_time: datetime.datetime | None = None, fhour: int | None = None,
                         model_label: str = "UpperAirCAMNet 3 km", synthetic: bool = False, notes: list[str] | None = None,
                         height_interval_dam: int = 12, div_smooth_km: float = 20.0, barb_spacing_km: float = 90.0,
                         show_convergence: bool = True, show_divergence: bool = True, dpi: int = 130) -> str:
    """Draw one chart. ``z`` (m), ``u``/``v`` (m/s, grid-relative) are 2-D [ny, nx] arrays on ``grid``."""
    global _BASEMAP
    fields = compute_map_fields(grid, z, u, v, div_smooth_km)
    xs, ys = grid.xy_1d()
    X, Y = np.meshgrid(xs, ys)
    aspect = grid.ny / grid.nx
    fig = plt.figure(figsize=(12, max(6.0, 12 * aspect) + 1.4))

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        globe = ccrs.Globe(ellipse=None, semimajor_axis=grid.radius, semiminor_axis=grid.radius)
        proj = ccrs.LambertConformal(central_longitude=grid.lov, central_latitude=grid.latin1,
                                     standard_parallels=(grid.latin1, grid.latin2), globe=globe)
        ax = fig.add_axes([0.05, 0.15, 0.90, 0.74], projection=proj)
        ax.set_extent([xs[0], xs[-1], ys[0], ys[-1]], crs=proj)
        kw = {"transform": proj}
        if _BASEMAP is None:
            _BASEMAP = _basemap_available()
        if _BASEMAP:
            ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="#555555", linewidth=0.6, zorder=3)
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#333333", linewidth=0.8, zorder=3)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#333333", linewidth=0.8, zorder=3)
        else:
            (notes := list(notes or [])).append("basemap unavailable (Natural Earth data not downloaded) - lat/lon grid only")
        import matplotlib.ticker as mticker
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.6, linestyle=":",
                          x_inline=False, y_inline=False, rotate_labels=False)
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 5))
        gl.ylocator = mticker.FixedLocator(np.arange(0, 90, 2))
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 8}
    except ImportError:
        ax = fig.add_axes([0.06, 0.15, 0.9, 0.74])
        ax.set_aspect("equal")
        kw = {}
        lat, lon = grid.latlon()
        cs = ax.contour(X, Y, lat, levels=np.arange(10, 70, 5), colors="gray", linewidths=0.4, linestyles=":")
        ax.clabel(cs, fmt="%dN", fontsize=7)
        cs = ax.contour(X, Y, lon, levels=np.arange(-140, -50, 5), colors="gray", linewidths=0.4, linestyles=":")
        ax.clabel(cs, fmt="%dW", fontsize=7)
        ax.set_xticks([]), ax.set_yticks([])
        (notes := list(notes or [])).append("cartopy not installed - no basemap")

    # shaded isotachs
    cmap = mcolors.ListedColormap(ISOTACH_COLORS)
    norm = mcolors.BoundaryNorm(ISOTACH_LEVELS, cmap.N)
    spd = np.where(fields["speed_kt"] >= ISOTACH_LEVELS[0], fields["speed_kt"], np.nan)
    shade = ax.contourf(X, Y, spd, levels=ISOTACH_LEVELS, cmap=cmap, norm=norm, extend="max", zorder=1, **kw)

    # height contours (dam)
    h = fields["height_dam"]
    lo, hi = np.floor(np.nanmin(h) / height_interval_dam) * height_interval_dam, np.nanmax(h) + height_interval_dam
    hc = ax.contour(X, Y, h, levels=np.arange(lo, hi, height_interval_dam), colors="black", linewidths=1.4, zorder=5, **kw)
    ax.clabel(hc, fmt="%d", fontsize=8, inline=True)

    # divergence / convergence
    d = fields["divergence_1e5"]
    if show_divergence:
        dc = ax.contour(X, Y, d, levels=DIVERGENCE_LEVELS, colors="#d4008f", linewidths=1.0, zorder=6, **kw)
        ax.clabel(dc, fmt="%d", fontsize=7, inline=True)
    if show_divergence and show_convergence:
        cc = ax.contour(X, Y, d, levels=[-x for x in DIVERGENCE_LEVELS[::-1]], colors="#1f4fd1", linewidths=0.9,
                        linestyles="--", zorder=6, **kw)
        ax.clabel(cc, fmt="%d", fontsize=7, inline=True)

    # wind barbs (kt), thinned
    step = max(1, int(round(barb_spacing_km * 1000.0 / grid.dx)))
    sl = (slice(step // 2, None, step), slice(step // 2, None, step))
    ax.barbs(X[sl], Y[sl], u[sl] * KT, v[sl] * KT, length=5.5, linewidth=0.6, color="#222222", zorder=7, **kw)

    cax = fig.add_axes([0.2, 0.055, 0.6, 0.016])
    cb = fig.colorbar(shade, cax=cax, orientation="horizontal")
    cb.set_label("Isotachs (kt)")

    div_title = (f"  |  Divergence (10$^{{-5}}$ s$^{{-1}}$, magenta{' / convergence dashed blue' if show_convergence else ''})"
                 if show_divergence else "")
    fig.text(0.03, 0.955, f"{level_hpa} mb  Heights (dam, black)  |  Isotachs (kt, shaded)  |  Wind (kt, barbs)" + div_title,
             fontsize=10.5, weight="bold")
    when = ""
    if init_time is not None and fhour is not None:
        valid = init_time + datetime.timedelta(hours=int(fhour))
        when = f"Init {init_time:%Y-%m-%d %HZ}   F{int(fhour):02d}   Valid {valid:%a %Y-%m-%d %HZ}"
    fig.text(0.03, 0.925, f"{model_label}   |   {grid.dx / 1000:.0f} km grid {grid.nx}x{grid.ny}   |   {when}", fontsize=9.5)
    if synthetic:
        fig.text(0.97, 0.985, "SYNTHETIC DATA - NOT A REAL FORECAST", color="red", ha="right", va="top", fontsize=10,
                 weight="bold")
    footer = ([f"divergence smoothed {div_smooth_km:g} km" if div_smooth_km else "divergence unsmoothed"]
              if show_divergence else []) + list(notes or [])
    fig.text(0.03, 0.012, "   |   ".join(footer), fontsize=7.5, color="#444444")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
