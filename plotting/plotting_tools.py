import math

from matplotlib.patches import Patch

from plotting.common import event_hatch, mean_std, series_label


def clustered_positions(nseries, nevents, cluster_width=0.82):
    if nevents <= 0:
        return [], 0.0
    width = cluster_width / nevents
    positions = {}
    for sidx in range(nseries):
        for eidx in range(nevents):
            positions[(sidx, eidx)] = sidx - cluster_width / 2 + width / 2 + eidx * width
    return positions, width


def event_legend(events):
    handles = []
    for events_value in events:
        exponent = round(math.log10(events_value))
        label = (
            rf"$10^{{{exponent}}}$ Events"
            if events_value == 10 ** exponent
            else f"{events_value:g}"
        )
        handles.append(
            Patch(
                facecolor="0.8",
                edgecolor="black",
                hatch=event_hatch(events_value, events),
                label=label,
            )
        )
    return handles


def stage_legend(stage_names, colors):
    return [Patch(facecolor=color, edgecolor="black", label=stage) for stage, color in zip(stage_names, colors)]


def plot_runtime_bars(ax, series, events, samples, ylabel, show_threads=True):
    positions, width = clustered_positions(len(series), len(events))
    for sidx, key in enumerate(series):
        for eidx, event_count in enumerate(events):
            values = samples.get((key, event_count), [])
            if not values:
                continue
            mean, std = mean_std(values)
            ax.bar(
                positions[(sidx, eidx)],
                mean,
                width=width * 0.92,
                color="0.75",
                edgecolor="black",
                hatch=event_hatch(event_count, events),
                yerr=std,
                capsize=2,
            )

    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([series_label(key, show_threads=show_threads) for key in series], rotation=25)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    return positions, width
