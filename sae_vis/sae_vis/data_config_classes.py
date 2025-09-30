from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator

from frozendict import frozendict
from rich import print as rprint
from rich.table import Table
from rich.tree import Tree

SEQUENCES_CONFIG_HELP = dict(
    buffer="How many tokens to add as context to each sequence, on each side. The tokens chosen for the top acts / \
quantile groups can't be outside the buffer range. If None, we use the entire sequence as context.",
    compute_buffer="If False, then we don't compute the loss effect, activations, or any other data for tokens \
other than the bold tokens in our sequences (saving time).",
    n_quantiles="Number of quantile groups for the sequences. If zero, we only show top activations, no quantile \
groups.",
    top_acts_group_size="Number of sequences in the 'top activating sequences' group.",
    quantile_group_size="Number of sequences in each of the sequence quantile groups.",
    top_logits_hoverdata="Number of top/bottom logits to show in the hoverdata for each token.",
    hover_below="Whether the hover information about a token appears below or above the token.",
    othello="If True, we make Othello boards instead of sequences (requires OthelloGPT)",
    n_boards_per_row="Only relevant for Othello, sets number of boards per row in top examples",
    dfa_for_attn_crosscoders="Only relevant for attention crosscoders. If true, shows DFA for top attention tokens.",
)

ACTIVATIONS_HISTOGRAM_CONFIG_HELP = dict(
    n_bins="Number of bins for the histogram.",
)

LOGITS_HISTOGRAM_CONFIG_HELP = dict(
    n_bins="Number of bins for the histogram.",
)

LOGITS_TABLE_CONFIG_HELP = dict(
    n_rows="Number of top/bottom logits to show in the table.",
)

LAYER_ACTIVATION_PLOT_CONFIG_HELP = dict(
    height="Height of the layer activation plot in pixels.",
    show_mean="Whether to show the mean activation line across tokens.",
)

PROBE_LOGITS_TABLE_CONFIG_HELP = dict(
    n_rows="Number of top/bottom logits to show in the table, for each probe.",
)

FEATURE_TABLES_CONFIG_HELP = dict(
    n_rows="Number of rows to show for each feature table.",
    neuron_alignment_table="Whether to show the neuron alignment table.",
    correlated_neurons_table="Whether to show the correlated neurons table.",
    correlated_features_table="Whether to show the (pairwise) correlated features table.",
    correlated_b_features_table="Whether to show the correlated encoder-B features table.",
)


@dataclass
class BaseComponentConfig:
    def data_is_contained_in(self, other: "BaseComponentConfig") -> bool:
        """
        This returns False only when the data that was computed based on `other` wouldn't be enough to show the data
        that was computed based on `self`. For instance, if `self` was a config object with 10 rows, and `other` had
        just 5 rows, then this would return False. A less obvious example: if `self` was a histogram config with 50 bins
        then `other` would need to have exactly 50 bins (because we can't change the bins after generating them).
        """
        return True

    @property
    def help_dict(self) -> dict[str, str]:
        """
        This is a dictionary which maps the name of each argument to a description of what it does. This is used when
        printing out the help for a config object, to show what each argument does.
        """
        return {}


@dataclass
class PromptConfig(BaseComponentConfig):
    pass


@dataclass
class SeqMultiGroupConfig(BaseComponentConfig):
    buffer: tuple[int, int] | None = (5, 5)
    compute_buffer: bool = True
    n_quantiles: int = 10
    top_acts_group_size: int = 40  # Increased for more sequences in top section
    quantile_group_size: int = 12  # Increased for more sequences per quantile
    top_logits_hoverdata: int = 5
    hover_below: bool = True

    # Everything for specific kinds of crosscoders / base models
    othello: bool = False
    n_boards_per_row: int = 3
    dfa_for_attn_crosscoders: bool = True
    dfa_buffer: tuple[int, int] | None = (5, 5)

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return all(
            [
                self.buffer is None
                or (
                    other.buffer is not None and self.buffer[0] <= other.buffer[0]
                ),  # the buffer needs to be <=
                self.buffer is None
                or (other.buffer is not None and self.buffer[1] <= other.buffer[1]),
                int(self.compute_buffer)
                <= int(
                    other.compute_buffer
                ),  # we can't compute the buffer if we didn't in `other`
                self.n_quantiles
                in [
                    0,
                    other.n_quantiles,
                ],  # we actually need the quantiles identical (or one to be zero)
                self.top_acts_group_size
                <= other.top_acts_group_size,  # group size needs to be <=
                self.quantile_group_size
                <= other.quantile_group_size,  # each quantile group needs to be <=
                self.top_logits_hoverdata
                <= other.top_logits_hoverdata,  # hoverdata rows need to be <=
            ]
        )

    def __post_init__(self):
        # Get list of group lengths, based on the config params
        self.group_sizes = [self.top_acts_group_size] + [
            self.quantile_group_size
        ] * self.n_quantiles

    @property
    def help_dict(self) -> dict[str, str]:
        return SEQUENCES_CONFIG_HELP


@dataclass
class ActsHistogramConfig(BaseComponentConfig):
    n_bins: int = 50

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_bins == other.n_bins

    @property
    def help_dict(self) -> dict[str, str]:
        return ACTIVATIONS_HISTOGRAM_CONFIG_HELP


@dataclass
class LogitsHistogramConfig(BaseComponentConfig):
    n_bins: int = 50

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_bins == other.n_bins

    @property
    def help_dict(self) -> dict[str, str]:
        return LOGITS_HISTOGRAM_CONFIG_HELP


@dataclass
class LayerActivationPlotConfig(BaseComponentConfig):
    height: int = 300
    show_mean: bool = True

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return True  # Layer activation plots don't have size constraints

    @property
    def help_dict(self) -> dict[str, str]:
        return LAYER_ACTIVATION_PLOT_CONFIG_HELP


@dataclass
class CrossLayerTrajectoryConfig(BaseComponentConfig):
    """Configuration for cross-layer feature decoder norm trajectory plot"""
    n_sequences: int = 1  # Not used for decoder norms (always shows single trajectory)
    height: int = 400  # Height of the plot
    normalize: bool = True  # Whether to normalize decoder norms to [0, 1]
    show_mean: bool = True  # Whether to show the trajectory line

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_sequences <= other.n_sequences

    @property
    def help_dict(self) -> dict[str, str]:
        return {
            "n_sequences": "Not used for decoder norm trajectory (always single trajectory)",
            "height": "Height of the trajectory plot",
            "normalize": "Whether to normalize decoder norms to [0, 1] range",
            "show_mean": "Whether to show the decoder norm trajectory line"
        }


@dataclass
class CrossLayerDecoderNormsConfig(BaseComponentConfig):
    """Configuration for decoder norms plot (Plot 1)"""
    pass


@dataclass
class CrossLayerActivationHeatmapConfig(BaseComponentConfig):
    """Configuration for activation heatmap (Plot 2)"""
    example_text: str = "The quick brown fox jumps over the lazy dog."

    @property
    def help_dict(self) -> dict[str, str]:
        return {
            "example_text": "Text to use for generating the activation heatmap"
        }


@dataclass
class CrossLayerAggregatedActivationConfig(BaseComponentConfig):
    """Configuration for aggregated activation profile (Plot 3)"""
    max_samples: int = 100
    batch_size: int = 8

    @property
    def help_dict(self) -> dict[str, str]:
        return {
            "max_samples": "Maximum number of samples to process for aggregation",
            "batch_size": "Batch size for processing samples"
        }


@dataclass
class CrossLayerDLAConfig(BaseComponentConfig):
    """Configuration for direct logit attribution (Plot 4)"""
    top_k: int = 10

    @property
    def help_dict(self) -> dict[str, str]:
        return {
            "top_k": "Number of top/bottom tokens to show per layer"
        }


@dataclass
class CrossLayerFeatureCorrelationConfig(BaseComponentConfig):
    """Configuration for feature correlation heatmap (Plot 5)"""
    n_features: int = 50
    max_samples: int = 50
    batch_size: int = 16

    @property
    def help_dict(self) -> dict[str, str]:
        return {
            "n_features": "Number of features to include in correlation matrix",
            "max_samples": "Maximum number of samples to process",
            "batch_size": "Batch size for processing samples"
        }


@dataclass
class DecoderNormCosineSimilarityConfig(BaseComponentConfig):
    """Configuration for decoder norm cosine similarity heatmap"""
    n_features: int = 50

    @property
    def help_dict(self) -> dict[str, str]:
        return {
            "n_features": "Number of features to include in cosine similarity matrix"
        }



@dataclass
class LogitsTableConfig(BaseComponentConfig):
    n_rows: int = 10

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_rows <= other.n_rows

    @property
    def help_dict(self) -> dict[str, str]:
        return LOGITS_TABLE_CONFIG_HELP


@dataclass
class ProbeLogitsTablesConfig(BaseComponentConfig):
    n_rows: int = 10
    othello: bool = False

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return self.n_rows <= other.n_rows

    @property
    def help_dict(self) -> dict[str, str]:
        return PROBE_LOGITS_TABLE_CONFIG_HELP


@dataclass
class FeatureTablesConfig(BaseComponentConfig):
    n_rows: int = 3
    neuron_alignment_table: bool = False
    correlated_neurons_table: bool = True
    correlated_features_table: bool = True
    correlated_b_features_table: bool = False

    def data_is_contained_in(self, other: BaseComponentConfig) -> bool:
        assert isinstance(other, self.__class__)
        return all(
            [
                self.n_rows <= other.n_rows,
                self.neuron_alignment_table <= other.neuron_alignment_table,
                self.correlated_neurons_table <= other.correlated_neurons_table,
                self.correlated_features_table <= other.correlated_features_table,
                self.correlated_b_features_table <= other.correlated_b_features_table,
            ]
        )

    @property
    def help_dict(self) -> dict[str, str]:
        return FEATURE_TABLES_CONFIG_HELP


GenericComponentConfig = (
    PromptConfig
    | SeqMultiGroupConfig
    | ActsHistogramConfig
    | LogitsHistogramConfig
    | LogitsTableConfig
    | ProbeLogitsTablesConfig
    | FeatureTablesConfig
    | LayerActivationPlotConfig
    | CrossLayerTrajectoryConfig
)


class Column:
    def __init__(
        self,
        *args: GenericComponentConfig,
        width: int | None = None,
    ):
        self.components = list(args)
        self.width = width

    def __iter__(self) -> Iterator[Any]:
        return iter(self.components)

    def __getitem__(self, idx: int) -> Any:
        return self.components[idx]

    def __len__(self) -> int:
        return len(self.components)


@dataclass
class CrosscoderVisLayoutConfig:
    """
    This object allows you to set all the ways the feature vis will be laid out.

    Args (specified by the user):
        columns:
            A list of `Column` objects, where each `Column` contains a list of component configs.
        height:
            The height of the vis (in pixels).

    Args (defined during __init__):
        seq_cfg: SeqMultiGroupConfig
            Contains all the parameters for the top activating sequences (and the
            quantile groups).
        act_hist_cfg: ActsHistogramConfig
            Contains all the parameters for the activations histogram.
        logits_hist_cfg: LogitsHistogramConfig
            Contains all the parameters for the logits histogram.
        logits_table_cfg: LogitsTableConfig
            Contains all the parameters for the logits table.
        probe_logits_table_cfg: ProbeLogitsTablesConfig
            Contains all the parameters for the probe logits table.
        feature_tables_cfg: FeatureTablesConfig
            Contains all the parameters for the feature tables.
        prompt_cfg: PromptConfig
            Contains all the parameters for the prompt-centric vis.
    """

    columns: dict[int, Column] = field(default_factory=dict)
    height: int = 750

    seq_cfg: SeqMultiGroupConfig | None = None
    act_hist_cfg: ActsHistogramConfig | None = None
    logits_hist_cfg: LogitsHistogramConfig | None = None
    logits_table_cfg: LogitsTableConfig | None = None
    probe_logits_table_cfg: ProbeLogitsTablesConfig | None = None
    feature_tables_cfg: FeatureTablesConfig | None = None
    layer_activation_plot_cfg: LayerActivationPlotConfig | None = None
    cross_layer_trajectory_cfg: CrossLayerTrajectoryConfig | None = None
    prompt_cfg: PromptConfig | None = None

    # New cross-layer visualization configs
    decoder_norms_cfg: CrossLayerDecoderNormsConfig | None = None
    activation_heatmap_cfg: CrossLayerActivationHeatmapConfig | None = None
    aggregated_activation_cfg: CrossLayerAggregatedActivationConfig | None = None
    dla_cfg: CrossLayerDLAConfig | None = None
    feature_correlation_cfg: CrossLayerFeatureCorrelationConfig | None = None
    decoder_norm_cosine_similarity_cfg: DecoderNormCosineSimilarityConfig | None = None

    COMPONENT_MAP: frozendict[str, str] = frozendict(
        {
            "Prompt": "prompt_cfg",
            "SeqMultiGroup": "seq_cfg",
            "ActsHistogram": "act_hist_cfg",
            "LogitsHistogram": "logits_hist_cfg",
            "LogitsTable": "logits_table_cfg",
            "ProbeLogitsTables": "probe_logits_table_cfg",
            "FeatureTables": "feature_tables_cfg",
            "LayerActivationPlot": "layer_activation_plot_cfg",
            "CrossLayerTrajectory": "cross_layer_trajectory_cfg",
            "CrossLayerDecoderNorms": "decoder_norms_cfg",
            "CrossLayerActivationHeatmap": "activation_heatmap_cfg",
            "CrossLayerAggregatedActivation": "aggregated_activation_cfg",
            "CrossLayerDLA": "dla_cfg",
            "CrossLayerFeatureCorrelation": "feature_correlation_cfg",
            "DecoderNormCosineSimilarity": "decoder_norm_cosine_similarity_cfg",
        }
    )

    def get_cfg_from_name(self, comp_name: str) -> Any:
        if comp_name in self.COMPONENT_MAP:
            return getattr(self, self.COMPONENT_MAP[comp_name])
        raise ValueError(f"Unknown component name {comp_name}")

    @property
    def components(self):
        """Returns a dictionary mapping component names (lowercase) to their configs, filtering out Nones."""
        all_components = {
            k[0].lower() + k[1:]: self.get_cfg_from_name(k) for k in self.COMPONENT_MAP
        }
        return {k: v for k, v in all_components.items() if v is not None}

    def __init__(self, columns: list[Column], height: int = 750):
        """
        The __init__ method will allow you to extract things like `self.seq_cfg` from the object (even though they're
        initially stored in the `columns` attribute). It also verifies that there are no duplicate components (which is
        redundant, and could mess up the HTML).
        """
        # Define the columns (as dict) and the height
        self.columns = {idx: col for idx, col in enumerate(columns)}
        self.height = height

        # Get a list of all our components, and verify there's no duplicates
        all_components = [
            component for column in self.columns.values() for component in column
        ]
        all_component_names = [
            comp.__class__.__name__.replace("Config", "") for comp in all_components
        ]
        assert len(all_component_names) == len(
            set(all_component_names)
        ), "Duplicate components in layout config"

        # Once we've verified this, store each config component as an attribute
        for comp, comp_name in zip(all_components, all_component_names):
            if comp_name in self.COMPONENT_MAP:
                setattr(self, self.COMPONENT_MAP[comp_name], comp)
            else:
                raise ValueError(f"Unknown component name {comp_name}")

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Returns string-ified METADATA, to be dumped into the JavaScript page. Fpr example, default
        Othello layout would return:

            {
                "layout": [["featureTables"], ["actsHistogram", "logitsTable"], ["seqMultiGroup"]],
                "othello": True,
            }
        """

        def config_name_to_component_name(config_name: str) -> str:
            component_name = config_name.replace("Config", "")
            component_name = component_name[0].lower() + component_name[1:]
            if component_name == "sequences":
                component_name = "seqMultiGroup"
            # Handle cross-layer components
            elif component_name == "crossLayerDecoderNorms":
                component_name = "decoderNorms"
            elif component_name == "crossLayerActivationHeatmap":
                component_name = "activationHeatmap"
            elif component_name == "crossLayerAggregatedActivation":
                component_name = "aggregatedActivation"
            elif component_name == "crossLayerDLA":
                component_name = "dla"
            elif component_name == "crossLayerFeatureCorrelation":
                component_name = "featureCorrelation"
            elif component_name == "decoderNormCosineSimilarity":
                component_name = "decoderNormCosineSimilarity"
            return component_name

        layout = [
            [config_name_to_component_name(comp.__class__.__name__) for comp in column]
            for column in self.columns.values()
        ]
        othello = self.seq_cfg.othello if (self.seq_cfg is not None) else False

        column_widths = [column.width for column in self.columns.values()]

        return {
            "layout": layout,
            "othello": othello,
            "columnWidths": column_widths,
            "height": self.height,
        }

    def data_is_contained_in(self, other: "CrosscoderVisLayoutConfig") -> bool:
        """
        Returns True if `self` uses only data that would already exist in `other`. This is useful because our prompt-
        centric vis needs to only use data that was already computed as part of our initial data gathering. For example,
        if our CrosscoderVisData object only contains the first 10 rows of the logits table, then we can't show the top 15 rows
        in the prompt centric view!
        """
        for comp_name, comp in self.components.items():
            # If the component in `self` is not present in `other`, return False
            if comp_name not in other.components:
                return False
            # If the component in `self` is present in `other`, but the `self` component is larger, then return False
            comp_other = other.components[comp_name]
            if not comp.data_is_contained_in(comp_other):
                return False

        return True

    def help(
        self,
        title: str = "CrosscoderVisLayoutConfig",
        key: bool = True,
    ) -> Tree | None:
        """
        This prints out a tree showing the layout of the vis, by column (as well as the values of the arguments for each
        config object, plus their default values if they changed, and the descriptions of each arg).
        """

        # Create tree (with title and optionally the key explaining arguments)
        if key:
            title += "\n\n" + KEY_LAYOUT_VIS
        tree = Tree(title)

        n_columns = len(self.columns)

        # For each column, add a tree node
        for column_idx, vis_components in self.columns.items():
            n_components = len(vis_components)
            tree_column = tree.add(f"Column {column_idx}")

            # For each component in that column, add a tree node
            for component_idx, vis_component in enumerate(vis_components):
                n_params = len(asdict(vis_component))
                tree_component = tree_column.add(
                    f"{vis_component.__class__.__name__}".rstrip("Config")
                )

                # For each config parameter of that component
                for param_idx, (param, value) in enumerate(
                    asdict(vis_component).items()
                ):
                    # Get line break if we're at the final parameter of this component (unless it's the final component
                    # in the final column)
                    suffix = "\n" if (param_idx == n_params - 1) else ""
                    if (component_idx == n_components - 1) and (
                        column_idx == n_columns - 1
                    ):
                        suffix = ""

                    # Get argument description, and its default value
                    desc = vis_component.help_dict.get(param, "")
                    value_default = getattr(
                        vis_component.__class__, param, "no default"
                    )

                    # Add tree node (appearance is different if value is changed from default)
                    if value != value_default:
                        info = f"[b dark_orange]{param}: {value!r}[/] (default = {value_default!r}) \n[i #888888]{desc}[/]{suffix}"
                    else:
                        info = f"[b #00aa00]{param}: {value!r}[/] \n[i #888888]{desc}[/]{suffix}"
                    tree_component.add(info)

        rprint(tree)

    @classmethod
    def default_feature_centric_layout(cls) -> "CrosscoderVisLayoutConfig":
        return cls(
            columns=[
                Column(
                    FeatureTablesConfig(),
                    ActsHistogramConfig(),
                    CrossLayerTrajectoryConfig(),
                    width=600  # Wider left column for both graphs
                ),
                Column(SeqMultiGroupConfig(), width=800),  # Sequences on the right
            ],
            height=800,
        )

    @classmethod
    def default_prompt_centric_layout(cls) -> "CrosscoderVisLayoutConfig":
        return cls(
            columns=[
                Column(
                    PromptConfig(),
                    ActsHistogramConfig(),
                    LogitsTableConfig(n_rows=5),
                    SeqMultiGroupConfig(top_acts_group_size=10, n_quantiles=0),
                    LogitsHistogramConfig(),
                    width=420,
                ),
            ],
            height=1100,
        )

    @classmethod
    def default_othello_layout(cls, boards: bool = True) -> "CrosscoderVisLayoutConfig":
        return cls(
            columns=[
                Column(FeatureTablesConfig()),
                Column(
                    ActsHistogramConfig(),
                    LogitsTableConfig(),
                    ProbeLogitsTablesConfig(),
                ),
                Column(
                    SeqMultiGroupConfig(
                        othello=boards,
                        buffer=None,
                        compute_buffer=not boards,
                        n_quantiles=5,
                        quantile_group_size=6,
                        top_acts_group_size=24,
                    ),
                ),
            ],
            height=1250,
        )


KEY_LAYOUT_VIS = """Key:
  the tree shows which components will be displayed in each column (from left to right)
  arguments are [b #00aa00]green[/]
  arguments changed from their default are [b dark_orange]orange[/], with default in brackets
  argument descriptions are in [i]italics[/i]
"""


CROSSCODER_CONFIG_DICT = dict(
    layers="The layers to use for the crosscoder analysis",
    features="The set of features which we'll be gathering data for. If an integer, we only get data for 1 feature",
    minibatch_size_tokens="The minibatch size we'll use to split up the full batch during forward passes, to avoid \
OOMs.",
    minibatch_size_features="The feature minibatch size we'll use to split up our features, to avoid OOM errors",
    seed="Random seed, for reproducibility (e.g. sampling quantiles)",
    verbose="Whether to print out progress messages and other info during the data gathering process",
)


@dataclass
class CrosscoderVisConfig:
    # Data
    features: int | Iterable[int] | None = None
    minibatch_size_features: int = 256
    minibatch_size_tokens: int = 64
    seqpos_slice: tuple[int | None, ...] = (None, None, None)

    # Vis
    feature_centric_layout: CrosscoderVisLayoutConfig = field(
        default_factory=CrosscoderVisLayoutConfig.default_feature_centric_layout
    )
    prompt_centric_layout: CrosscoderVisLayoutConfig = field(
        default_factory=CrosscoderVisLayoutConfig.default_prompt_centric_layout
    )

    # Misc
    seed: int | None = 0
    verbose: bool = False

    # Depreciated
    batch_size: None = None

    def __post_init__(self):
        assert (
            self.batch_size is None
        ), "The `batch_size` parameter has been depreciated. Please use `minibatch_size_tokens` instead."
        assert (
            len(self.prompt_centric_layout.columns) == 1
        ), "Only allowed a single column for prompt-centric layout."

    def help(self, title: str = "CrosscoderVisConfig"):
        """
        Performs the `help` method for both of the layout objects, as well as for the non-layout-based configs.
        """
        # Create table for all the non-layout-based params
        table = Table(
            "Param", "Value (default)", "Description", title=title, show_lines=True
        )

        # Populate table (middle row is formatted based on whether value has changed from default)
        for param, desc in CROSSCODER_CONFIG_DICT.items():
            value = getattr(self, param)
            value_default = getattr(self.__class__, param, "no default")
            if value != value_default:
                value_default_repr = (
                    "no default"
                    if value_default == "no default"
                    else repr(value_default)
                )
                value_str = f"[b dark_orange]{value!r}[/]\n({value_default_repr})"
            else:
                value_str = f"[b #00aa00]{value!r}[/]"
            table.add_row(param, value_str, f"[i]{desc}[/]")

        # Print table, and print the help trees for the layout objects
        rprint(table)
        self.feature_centric_layout.help(
            title="CrosscoderVisLayoutConfig: feature-centric vis", key=False
        )
        self.prompt_centric_layout.help(
            title="CrosscoderVisLayoutConfig: prompt-centric vis", key=False
        )
